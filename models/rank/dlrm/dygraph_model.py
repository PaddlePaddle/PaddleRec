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
        dense_feature_dim = config.get("hyper_parameters.dense_input_dim")
        bot_layer_sizes = config.get("hyper_parameters.bot_layer_sizes")
        sparse_feature_number = config.get(
            "hyper_parameters.sparse_feature_number")
        sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
        top_layer_sizes = config.get("hyper_parameters.top_layer_sizes")
        num_field = config.get("hyper_parameters.num_field")

        dlrm_model = net.DLRMLayer(
            dense_feature_dim=dense_feature_dim,
            bot_layer_sizes=bot_layer_sizes,
            sparse_feature_number=sparse_feature_number,
            sparse_feature_dim=sparse_feature_dim,
            top_layer_sizes=top_layer_sizes,
            num_field=num_field,
            self_interaction=False)

        return dlrm_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
        sparse_tensor = []
        for b in batch_data[:-1]:
            sparse_tensor.append(
                paddle.to_tensor(b.numpy().astype('int64').reshape(-1, 1)))
        dense_tensor = paddle.to_tensor(batch_data[-1].numpy().astype(
            'float32').reshape(-1, dense_feature_dim))
        label = sparse_tensor[0]
        return label, sparse_tensor[1:], dense_tensor

    # define loss function by predicts and label
    def create_loss(self, raw_predict_2d, label):
        cost = paddle.nn.functional.cross_entropy(
            input=raw_predict_2d, label=label)
        avg_cost = paddle.mean(x=cost)
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
        metrics_list_name = ["auc", "accuracy"]
        auc_metric = paddle.metric.Auc("ROC")
        accuracy_metric = paddle.metric.Accuracy()
        metrics_list = [auc_metric, accuracy_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        label, sparse_tensor, dense_tensor = self.create_feeds(batch_data,
                                                               config)

        raw_pred_2d = dy_model.forward(sparse_tensor, dense_tensor)
        loss = self.create_loss(raw_pred_2d, label)
        # update metrics
        predict_2d = paddle.nn.functional.softmax(raw_pred_2d)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        metrics_list[1].update(metrics_list[1].compute(
            pred=predict_2d, label=label).numpy())

        # print_dict format :{'loss': loss}
        print_dict = {"loss": loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        label, sparse_tensor, dense_tensor = self.create_feeds(batch_data,
                                                               config)

        raw_pred_2d = dy_model.forward(sparse_tensor, dense_tensor)
        # update metrics
        predict_2d = paddle.nn.functional.softmax(raw_pred_2d)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        metrics_list[1].update(metrics_list[1].compute(
            pred=predict_2d, label=label).numpy())
        return metrics_list, None
