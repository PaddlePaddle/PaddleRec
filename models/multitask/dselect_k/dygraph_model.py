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
import net


class DygraphModel():
    # define model
    def create_model(self, config):
        feature_size = config.get('hyper_parameters.feature_size', None)
        expert_num = config.get('hyper_parameters.expert_num', None)
        expert_size = config.get('hyper_parameters.expert_size', None)
        tower_size = config.get('hyper_parameters.tower_size', None)
        gate_num = config.get('hyper_parameters.gate_num', None)
        top_k = config.get('hyper_parameters.top_k', 2)

        MoE = net.MoELayer(feature_size, expert_num, expert_size, tower_size,
                           gate_num, top_k)

        return MoE

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        feature_size = config.get('hyper_parameters.feature_size', None)
        input_data = paddle.to_tensor(batch_data[0].numpy().astype('float32')
                                      .reshape(-1, 1, 36, 36))
        label_left = paddle.to_tensor(batch_data[1].numpy().astype("int64")
                                      .reshape(-1, 1))
        label_right = paddle.to_tensor(batch_data[2].numpy().astype("int64")
                                       .reshape(-1, 1))
        return input_data, label_left, label_right

    # define loss function by predicts and label
    def create_loss(self, pred_left, pred_right, label_left, label_right):
        cost_left = paddle.nn.functional.cross_entropy(
            input=pred_left, label=label_left)
        cost_right = paddle.nn.functional.cross_entropy(
            input=pred_right, label=label_right)
        cost = cost_left + cost_right

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
        metrics_list_name = ["acc_left", "acc_right"]
        acc_left_metric = paddle.metric.Accuracy()
        acc_right_metric = paddle.metric.Accuracy()
        metrics_list = [acc_left_metric, acc_right_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        input_data, label_left, label_right = self.create_feeds(batch_data,
                                                                config)
        pred_left, pred_right = dy_model.forward(input_data)
        loss = self.create_loss(pred_left, pred_right, label_left, label_right)

        # update metrics
        metrics_list[0].update(metrics_list[0].compute(
            pred=pred_left, label=label_left).numpy())
        metrics_list[1].update(metrics_list[1].compute(
            pred=pred_right, label=label_right).numpy())

        print_dict = {'loss': loss}
        # print_dict = None
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        input_data, label_left, label_right = self.create_feeds(batch_data,
                                                                config)
        pred_left, pred_right = dy_model.forward(input_data)

        # update metrics
        metrics_list[0].update(metrics_list[0].compute(
            pred=pred_left, label=label_left).numpy())
        metrics_list[1].update(metrics_list[1].compute(
            pred=pred_right, label=label_right).numpy())

        return metrics_list, None
