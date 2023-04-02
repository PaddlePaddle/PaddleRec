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
        init_value = 0.1
        sparse_feature_number = config.get(
            'hyper_parameters.sparse_feature_number', None)
        reg = config.get('hyper_parameters.reg', None)
        num_field = config.get('hyper_parameters.num_field', None)

        LR = net.LRLayer(sparse_feature_number, init_value, reg, num_field)
        return LR

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        num_field = config.get('hyper_parameters.num_field', None)
        label = paddle.to_tensor(batch_data[0].numpy().astype('int64').reshape(
            -1, 1))
        feat_idx = paddle.to_tensor(batch_data[1].numpy().astype('int64')
                                    .reshape(-1, num_field))
        raw_feat_value = paddle.to_tensor(batch_data[2].numpy().astype(
            'float32').reshape(-1, num_field))
        feat_value = paddle.reshape(raw_feat_value,
                                    [-1, num_field])  # None * num_field * 1
        return label, feat_idx, feat_value

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                label, dtype="float32"))
        avg_cost = paddle.sum(x=cost)
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
        metrics_list_name = ["auc"]
        auc_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        label, feat_idx, feat_value = self.create_feeds(batch_data, config)
        pred = dy_model.forward(feat_idx, feat_value)

        loss = self.create_loss(pred, label)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        # print_dict format :{'loss': loss}
        print_dict = None
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        label, feat_idx, feat_value = self.create_feeds(batch_data, config)
        pred = dy_model.forward(feat_idx, feat_value)

        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        return metrics_list, None
