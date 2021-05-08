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
import os
import paddle.nn as nn
import paddle.nn.functional as F
import math

from net import DeepRetrieval
from paddle.distributed.fleet.dataset.index_dataset import GraphIndex


class DygraphModel():
    # define model
    def save_item_path(self, path, prefix):
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, str(prefix))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, self.path_save_file_name)
        self.graph_index._graph.save(path)     

    def load_item_path(self, path):
        path = os.path.join(path, self.path_save_file_name)
        self.graph_index._graph.load(path)

    def create_model(self, config):
        print("create model")
        self.path_save_file_name = "path_save";
        self.width = config.get("hyper_parameters.width")
        self.height = config.get("hyper_parameters.height")
        self.beam_search_num = config.get(
            "hyper_parameters.beam_search_num")
        self.item_path_volume = config.get(
            "hyper_parameters.item_path_volume")
        self.user_embedding_size = config.get(
            "hyper_parameters.user_embedding_size")
        self.graph_index = GraphIndex(
            "test", self.width, self.height, self.item_path_volume)

        init_model_path = config.get("model_init_path")

        if init_model_path == None:
            self.graph_index._init_by_random()
        else:
            self.graph_index._init_graph(os.path.join(init_model_path, self.path_save_file_name))
        self.use_multi_task_learning = config.get("hyper_parameters.use_multi_task_learning")
        self.item_count = config.get("hyper_parameters.item_count")
        if self.use_multi_task_learning:
            self.multi_task_layer_size = config.get("hyper_parameters.multi_task_layer_size")
        else:
            self.multi_task_layer_size = None
        dr_model = DeepRetrieval(self.width, self.height, self.beam_search_num,
                                 self.item_path_volume, self.user_embedding_size, self.item_count, self.use_multi_task_learning, self.multi_task_layer_size)

        return dr_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        print("batch_data ",batch_data, "len_batch_data",len(batch_data))
        # user_embedding = paddle.to_tensor(batch_data[0].numpy().astype(
        #     'float32').reshape(-1, self.user_embedding_size))

        # item_id = batch_data[1]  # (batch_size, 1)   if use_multi_task (batch_size,3)
        # item_id = item_id.numpy().tolist()
        # item_path_id = []
        # multi_task_pos_label = []
        # multi_task_neg_label = []
        # for i in item_id:
        #     item_path_id.append(self.graph_index.get_path_of_item(i[0]))
        #     if self.use_multi_task_learning:
        #         multi_task_pos_label.append([i[0]])
        #         multi_task_neg_label.append([i[1]])


        # item_path_kd_label = []

        # # for every example
        # for item_path in item_path_id:
        #     item_kd_represent = []
        #     # for every path choice of item
        #     for item_path_j in item_path[0]:
        #         path_label = np.array(
        #             self.graph_index.path_id_to_kd_represent(item_path_j))
        #         #item_kd_represent.append(paddle.to_tensor(
        #         #    path_label.astype('int64').reshape(1, self.width)))
        #         item_kd_represent.append(paddle.to_tensor(
        #            path_label.astype('int32').reshape(self.width)))
        #     item_path_kd_label.append(paddle.reshape(paddle.concat(item_kd_represent,axis=-1),[-1, self.width]))
        # #print("label shape ",item_path_kd_label.shape)    
        # item_path_kd_label=paddle.concat(item_path_kd_label, axis = 0)

        
        if self.use_multi_task_learning:
            return batch_data[0:5]
        # return batch[0:2]
        #     #return user_embedding, item_path_kd_label, multi_task_pos_label ,   multi_task_neg_label
        #     multi_task_pos_label = paddle.to_tensor(multi_task_pos_label)
        #     multi_task_neg_label = paddle.to_tensor(multi_task_neg_label)
        item_id, user_embedding, item_path_kd_label = batch_data[0:3]
        #return user_embedding, item_path_kd_label, multi_task_pos_label ,   multi_task_neg_label
        return  item_id, user_embedding, item_path_kd_label, None, None

    def create_infer_feeds(self, batch_data, config):
        user_embedding = paddle.to_tensor(batch_data[0].numpy().astype(
            'float32').reshape(-1, self.user_embedding_size))
        return user_embedding

    # define loss function by predicts and label
    def create_loss(self, path_prob, multi_task_loss):
        # path_prob: (batch_size * J, D)
        path_prob = paddle.prod(
            path_prob, axis=1, keepdim=True)  # (batch_size* J, 1)
        item_path_prob = paddle.reshape(
            path_prob, (-1, self.item_path_volume))  # (batch_size, J)
        item_path_prob = paddle.sum(item_path_prob, axis=1, keepdim=True)
        item_path_prob_log = paddle.log(item_path_prob)
        cost = -1 * paddle.sum(item_path_prob_log)
        print("cost: {}".format(cost))
        if self.use_multi_task_learning:
            cost = cost + multi_task_loss
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
        
        item_id, user_embedding, item_path_kd_label,multi_task_pos_label,multi_task_neg_label = self.create_feeds(batch_data,
                                                               config)


        path_prob,multi_task_loss = dy_model.forward(
            user_embedding,item_path_kd_label,multi_task_pos_label,multi_task_neg_label)

        loss = self.create_loss(path_prob,multi_task_loss)

        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        user_embedding = self.create_infer_feeds(batch_data,
                                                 config)

        kd_path, path_prob = dy_model.forward(user_embedding, is_infer=True)
        kd_path_list = kd_path.numpy().tolist()

        em_dict = {}
        # item = 0
        em_dict[0] = []
        for batch_idx, batch in enumerate(kd_path_list):
            for path_idx, path in enumerate(batch):
                path_id = self.graph_index.kd_represent_to_path_id(path)
                # em_dict[0].append("{}:{}".format(path_id, prob))

        return metrics_list, None
