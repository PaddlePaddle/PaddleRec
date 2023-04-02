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
        feature_size = config.get('hyper_parameters.feature_size', None)
        expert_num = config.get('hyper_parameters.expert_num', None)
        expert_size = config.get('hyper_parameters.expert_size', None)
        tower_size = config.get('hyper_parameters.tower_size', None)
        gate_num = config.get('hyper_parameters.gate_num', None)

        MMoE = net.MMoELayer(feature_size, expert_num, expert_size, tower_size,
                             gate_num)

        return MMoE

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        feature_size = config.get('hyper_parameters.feature_size', None)
        input_data = paddle.to_tensor(batch_data[0].numpy().astype('float32')
                                      .reshape(-1, feature_size))
        label_income = paddle.to_tensor(batch_data[1].numpy().astype('float32')
                                        .reshape(-1, 1))
        label_marital = paddle.to_tensor(batch_data[2].numpy().astype(
            'float32').reshape(-1, 1))
        return input_data, label_income, label_marital

    # define loss function by predicts and label
    def create_loss(self, pred_income, pred_marital, label_income,
                    label_marital):
        pred_income_1d = paddle.slice(
            pred_income, axes=[1], starts=[1], ends=[2])
        pred_marital_1d = paddle.slice(
            pred_marital, axes=[1], starts=[1], ends=[2])
        cost_income = paddle.nn.functional.log_loss(
            input=pred_income_1d, label=label_income)
        cost_marital = paddle.nn.functional.log_loss(
            input=pred_marital_1d, label=label_marital)

        avg_cost_income = paddle.mean(x=cost_income)
        avg_cost_marital = paddle.mean(x=cost_marital)

        cost = avg_cost_income + avg_cost_marital
        return cost

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["auc_income", "auc_marital"]
        auc_income_metric = paddle.metric.Auc("ROC")
        auc_marital_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_income_metric, auc_marital_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        input_data, label_income, label_marital = self.create_feeds(batch_data,
                                                                    config)
        pred_income, pred_marital = dy_model.forward(input_data)
        loss = self.create_loss(pred_income, pred_marital, label_income,
                                label_marital)

        # update metrics
        metrics_list[0].update(
            preds=pred_income.numpy(), labels=label_income.numpy())
        metrics_list[1].update(
            preds=pred_marital.numpy(), labels=label_marital.numpy())

        print_dict = {'loss': loss}
        # print_dict = None
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        input_data, label_income, label_marital = self.create_feeds(batch_data,
                                                                    config)
        pred_income, pred_marital = dy_model.forward(input_data)

        # update metrics
        metrics_list[0].update(
            preds=pred_income.numpy(), labels=label_income.numpy())
        metrics_list[1].update(
            preds=pred_marital.numpy(), labels=label_marital.numpy())
        return metrics_list, None
