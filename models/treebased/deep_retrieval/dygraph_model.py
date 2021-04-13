# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

from net import DeepRetrieval
from paddle.distributed.fleet.dataset.index_dataset import GraphIndex


class DygraphModel():
    # define model
    def create_model(self, config):
        self.width = config.get("hyper_parameters.width")
        self.height = config.get("hyper_parameters.height")
        self.beam_search_num = config.get(
            "hyper_parameters.beam_search_num")
        self.item_path_volume = config.get(
            "hyper_parameters.item_path_volume")
        self.user_embedding_size = config.get(
            "hyper_parameters.user_embedding_size")

        self.item_input_file = config.get(
            "runner.item_input_file")
        self.item_output_proto = config.get(
            "runner.item_output_proto")

        self.graph_index = GraphIndex(
            "test", self.width, self.height, self.item_path_volume)
        self.graph_index._init_by_random(
            self.item_input_file, self.item_output_proto)

        dr_model = DeepRetrieval(self.width, self.height, self.beam_search_num,
                                 self.item_path_volume, self.user_embedding_size)

        return dr_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        user_embedding = paddle.to_tensor(batch_data[0].numpy().astype(
            'float32').reshape(-1, self.user_embedding_size))

        item_id = batch_data[1]  # (batch_size, 1)
        item_id = item_id.numpy().tolist()
        item_path_id = []
        for i in item_id:
            item_path_id.append(self.graph_index.get_path_of_item(i))

        item_path_kd_label = []
        print("item_path_id: {}".format(item_path_id))

        # for every example
        for item_path in item_path_id:
            item_kd_represent = []

            print("item_path: {}".format(item_path[0]))
            # for every path choice of item
            for item_path_j in item_path[0]:
                path_label = np.array(
                    self.graph_index.path_id_to_kd_represent(item_path_j))
                item_kd_represent.append(paddle.to_tensor(
                    path_label.astype('int64').reshape(1, self.width)))

            item_path_kd_label.append(item_kd_represent)
        print("item_path_kd_label: {}".format(item_path_kd_label))
        return user_embedding, item_path_kd_label

    def create_infer_feeds(self, batch_data, config):
        user_embedding = paddle.to_tensor(batch_data[0].numpy().astype(
            'float32').reshape(-1, self.user_embedding_size))
        return user_embedding

    # define loss function by predicts and label
    def create_loss(self, layer_prob_output, item_path_kd_label):
        # layer_prob_output: list[ (batch_size, K), ... , (batch_size, K)]
        # item_path_kd_label: list [ list[ (1, D), ..., (1, D) ],..., ]
        kd_label_list = []
        for idx, all_kd_val in enumerate(item_path_kd_label):
            kd_label_list.append(paddle.concat(all_kd_val, axis=0))  # (J, D)
        kd_label = paddle.concat(kd_label_list, axis=0)  # (batch_size * J, D)
        print("kd_label: {}".format(kd_label))
        print("layer_prob_output: {}".format(layer_prob_output))

        for idx, val in enumerate(layer_prob_output):
            # (batch_size * J, K)
            layer_prob_output[idx] = paddle.unsqueeze(
                layer_prob_output[idx], axis=0)
            layer_prob_output[idx] = paddle.expand(layer_prob_output[idx], shape=[self.item_path_volume,
                                                                                  layer_prob_output[idx].shape[1], layer_prob_output[idx].shape[2]])
            layer_prob_output[idx] = paddle.reshape(layer_prob_output[idx], (
                self.item_path_volume * layer_prob_output[idx].shape[1], layer_prob_output[idx].shape[2]))

        selected_prob = []
        for idx in range(self.width):  # D
            # find prob of every layer
            print("layer_prob_output[{}]: {}".format(
                idx, layer_prob_output[idx]))
            cur_prob_idx = paddle.slice(
                kd_label, axes=[1], starts=[idx], ends=[idx+1])  # (batch_size * J, 1)
            print("cur_prob_idx: {}".format(cur_prob_idx))
            cur_prob = paddle.index_sample(
                layer_prob_output[idx], cur_prob_idx)  # (batch_size * J, 1)
            print("cur_prob: {}".format(cur_prob))
            selected_prob.append(cur_prob)

        path_prob = paddle.concat(selected_prob, axis=1)  # (batch_size * J, D)
        path_prob = paddle.prod(
            path_prob, axis=1, keepdim=True)  # (batch_size* J, 1)

        item_path_prob = paddle.reshape(
            path_prob, (-1, self.item_path_volume))
        item_path_prob = paddle.sum(item_path_prob, axis=1, keepdim=True)
        item_path_prob_log = paddle.log(item_path_prob)
        cost = -1 * paddle.sum(item_path_prob_log)
        print("cost: {}".format(cost))
        return cost

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    def create_metrics(self):
        metrics_list_name = []
        metrics_list = []
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        user_embedding, item_path_kd_label = self.create_feeds(batch_data,
                                                               config)

        layer_prob_output = dy_model.forward(user_embedding)

        loss = self.create_loss(layer_prob_output, item_path_kd_label)

        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        user_embedding = self.create_infer_feeds(batch_data,
                                                 config)

        kd_path_concat, kd_prob = dy_model.forward(user_embedding, True)
        kd_path_list = kd_path_concat.numpy().tolist()
        kd_prob_list = kd_prob.numpy().tolist()

        em_dict = {}
        # item = 0
        em_dict[0] = []
        for batch_idx, batch in enumerate(kd_path_list):
            for path_idx, path in enumerate(batch):
                path_id = self.graph_index.kd_represent_to_path_id(path)
                prob = kd_prob_list[batch_idx][path_idx]
                em_dict[0].append("{}:{}".format(path_id, prob))
        print(em_dict)
        self.graph_index.update_Jpath_of_item(
            em_dict,  T=3, J=self.item_path_volume, lamd=1e-7, factor=2)
        print("get_path_of_item(0): {}".format(
            self.graph_index.get_path_of_item(0)))
        print("get_item_of_path(0): {}".format(
            self.graph_index.get_item_of_path(0)))

        return metrics_list, None
