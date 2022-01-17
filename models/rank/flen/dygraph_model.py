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

from paddle.regularizer import L2Decay
import net


class DygraphModel():
    # define model
    def create_model(self, config):
        sparse_feature_number = config.get(
            "hyper_parameters.sparse_feature_number")
        sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
        fc_sizes = config.get("hyper_parameters.fc_sizes")
        layer_sizes_dnn = config.get("hyper_parameters.layer_sizes_dnn")
        sparse_num_field = config.get("hyper_parameters.sparse_num_field")
        sparse_inputs_slots = config.get(
            'hyper_parameters.sparse_inputs_slots')

        flen_model = net.FLENLayer(sparse_feature_number, sparse_feature_dim,
                                   sparse_inputs_slots, sparse_num_field,
                                   layer_sizes_dnn)

        return flen_model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, batch_data, config):
        sparse_tensor = []
        for b in batch_data:
            sparse_tensor.append(
                paddle.to_tensor(b.numpy().astype('int64').reshape(-1, 1)))

        label = sparse_tensor[-1]
        return label, sparse_tensor[:-1]

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        cost = paddle.nn.functional.binary_cross_entropy(
            # cost = paddle.nn.functional.log_loss(
            input=pred,
            label=paddle.cast(
                label, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        return avg_cost

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        # optimizer = paddle.optimizer.Adam(
        #     learning_rate=lr, parameters=dy_model.parameters(), lazy_mode=True)

        optimizer = paddle.optimizer.Adagrad(
            learning_rate=lr,
            epsilon=1e-06,
            parameters=dy_model.parameters(),
            initial_accumulator_value=1e-3)
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["auc"]
        auc_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        label, sparse_tensor = self.create_feeds(batch_data, config)

        pred = dy_model.forward(sparse_tensor)
        loss = self.create_loss(pred, label)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        # print_dict format :{'loss': loss} 
        # print_dict = None
        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        label, sparse_tensor = self.create_feeds(batch_data, config)

        pred = dy_model.forward(sparse_tensor)

        loss = self.create_loss(pred, label)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        print_dict = {'logloss': loss}
        return metrics_list, print_dict  #, None
