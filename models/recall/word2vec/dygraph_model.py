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
        sparse_feature_number = config.get(
            "hyper_parameters.sparse_feature_number")
        sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
        neg_num = config.get("hyper_parameters.neg_num")

        word2vec_model = net.Word2VecLayer(
            sparse_feature_number,
            sparse_feature_dim,
            neg_num,
            emb_name="emb",
            emb_w_name="emb_w",
            emb_b_name="emb_b")

        return word2vec_model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, batch_data, config):
        neg_num = config.get("hyper_parameters.neg_num")
        input_word = paddle.to_tensor(batch_data[0].numpy().astype('int64')
                                      .reshape(-1, 1))
        true_word = paddle.to_tensor(batch_data[1].numpy().astype('int64')
                                     .reshape(-1, 1))
        neg_word = paddle.to_tensor(batch_data[2].numpy().astype('int64')
                                    .reshape(-1, neg_num))
        return input_word, true_word, neg_word

    # define loss function by predicts and label
    def create_loss(self, true_logits, neg_logits, config):
        neg_num = config.get("hyper_parameters.neg_num")
        label_ones = paddle.full(
            shape=[paddle.shape(true_logits)[0], 1], fill_value=1.0)
        label_zeros = paddle.full(
            shape=[paddle.shape(true_logits)[0], neg_num], fill_value=0.0)

        true_logits = paddle.nn.functional.sigmoid(true_logits)
        true_xent = paddle.nn.functional.binary_cross_entropy(true_logits,
                                                              label_ones)
        neg_logits = paddle.nn.functional.sigmoid(neg_logits)
        neg_xent = paddle.nn.functional.binary_cross_entropy(neg_logits,
                                                             label_zeros)
        cost = paddle.add(true_xent, neg_xent)
        avg_cost = paddle.mean(x=cost)

        return avg_cost

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.SGD(learning_rate=lr,
                                         parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = []
        metrics_list = []
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        input_word, true_word, neg_word = self.create_feeds(batch_data, config)

        true_logits, neg_logits = dy_model.forward(
            [input_word, true_word, neg_word])
        loss = self.create_loss(true_logits, neg_logits, config)
        # print_dict format :{'loss': loss} 
        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict
