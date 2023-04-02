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
        fc_sizes = config.get("hyper_parameters.fc_sizes")

        rank_model = net.DNNLayer(sparse_feature_number, sparse_feature_dim,
                                  fc_sizes)
        return rank_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data):
        user_sparse_inputs = [
            paddle.to_tensor(batch_data[i].numpy().astype('int64').reshape(-1,
                                                                           1))
            for i in range(4)
        ]

        mov_sparse_inputs = [
            paddle.to_tensor(batch_data[4].numpy().astype('int64').reshape(
                -1, 1)), paddle.to_tensor(batch_data[5].numpy().astype(
                    'int64').reshape(-1, 4)), paddle.to_tensor(batch_data[
                        6].numpy().astype('int64').reshape(-1, 3))
        ]

        label_input = paddle.to_tensor(batch_data[7].numpy().astype('int64')
                                       .reshape(-1, 1))

        return user_sparse_inputs, mov_sparse_inputs, label_input

    # define loss function by predicts and label
    def create_loss(self, predict, label_input):
        cost = F.square_error_cost(
            predict, paddle.cast(
                x=label_input, dtype='float32'))
        avg_cost = paddle.mean(cost)
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

        batch_size = config.get("runner.train_batch_size", 128)
        user_sparse_inputs, mov_sparse_inputs, label_input = self.create_feeds(
            batch_data)

        predict = dy_model.forward(batch_size, user_sparse_inputs,
                                   mov_sparse_inputs, label_input)
        loss = self.create_loss(predict, label_input)
        # update metrics
        print_dict = {"loss": loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        batch_runner_result = {}
        batch_size = config.get("runner.infer_batch_size", 128)
        user_sparse_inputs, mov_sparse_inputs, label_input = self.create_feeds(
            batch_data)

        predict = dy_model.forward(batch_size, user_sparse_inputs,
                                   mov_sparse_inputs, label_input)
        # update metrics
        uid = user_sparse_inputs[0]
        movieid = mov_sparse_inputs[0]
        label = label_input
        predict = predict

        batch_runner_result["userid"] = uid.numpy().tolist()
        batch_runner_result["movieid"] = movieid.numpy().tolist()
        batch_runner_result["label"] = label.numpy().tolist()
        batch_runner_result["predict"] = predict.numpy().tolist()

        print_dict = {"predict": predict}
        return metrics_list, print_dict, batch_runner_result
