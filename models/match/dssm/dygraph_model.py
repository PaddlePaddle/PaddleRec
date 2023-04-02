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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import net


class DygraphModel():
    # define model
    def create_model(self, config):
        trigram_d = config.get('hyper_parameters.trigram_d', None)
        neg_num = config.get('hyper_parameters.neg_num', None)
        slice_end = config.get('hyper_parameters.slice_end', None)
        fc_sizes = config.get('hyper_parameters.fc_sizes', None)
        fc_acts = config.get('hyper_parameters.fc_acts', None)

        DSSM_model = net.DSSMLayer(trigram_d, neg_num, slice_end, fc_sizes,
                                   fc_acts)
        return DSSM_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds_train(self, batch_data, trigram_d):
        query = paddle.to_tensor(batch_data[0].numpy().astype('float32')
                                 .reshape(-1, trigram_d))
        doc_pos = paddle.to_tensor(batch_data[1].numpy().astype('float32')
                                   .reshape(-1, trigram_d))
        doc_negs = []
        for ele in batch_data[2:]:
            doc_negs.append(
                paddle.to_tensor(ele.numpy().astype('float32').reshape(
                    -1, trigram_d)))
        return [query, doc_pos] + doc_negs

    def create_feeds_infer(self, batch_data, trigram_d):
        query = paddle.to_tensor(batch_data[0].numpy().astype('float32')
                                 .reshape(-1, trigram_d))
        doc_pos = paddle.to_tensor(batch_data[1].numpy().astype('float32')
                                   .reshape(-1, trigram_d))
        return [query, doc_pos]

    # define loss function by predicts and label
    def create_loss(self, hit_prob):
        loss = -paddle.sum(paddle.log(hit_prob), axis=-1)
        avg_cost = paddle.mean(x=loss)
        return avg_cost

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = []
        metrics_list = []
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        trigram_d = config.get('hyper_parameters.trigram_d', None)
        inputs = self.create_feeds_train(batch_data, trigram_d)

        R_Q_D_p, hit_prob = dy_model.forward(inputs, False)
        loss = self.create_loss(hit_prob)
        # update metrics
        print_dict = {"loss": loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        trigram_d = config.get('hyper_parameters.trigram_d', None)
        inputs = self.create_feeds_infer(batch_data, trigram_d)

        R_Q_D_p, hit_prob = dy_model.forward(inputs, True)
        # update metrics
        print_dict = {"query_doc_sim": R_Q_D_p}
        return metrics_list, print_dict
