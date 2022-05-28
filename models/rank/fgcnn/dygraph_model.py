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
import net
import numpy as np


class DygraphModel():
    # define model
    def create_model(self, config):
        sparse_input_slot = config.get('hyper_parameters.sparse_inputs_slots')
        dense_input_slot = config.get('hyper_parameters.dense_inputs_slots')
        sparse_feature_size = config.get("hyper_parameters.sparse_feature_size")
        feature_name = config.get("hyper_parameters.feature_name")
        feature_dim = config.get("hyper_parameters.feature_dim", 20)
        conv_kernel_width = config.get("hyper_parameters.conv_kernel_width", (7, 7, 7, 7))
        conv_filters = config.get("hyper_parameters.conv_filters", (14, 16, 18, 20))
        new_maps = config.get("hyper_parameters.new_maps", (3, 3, 3, 3))
        pooling_width = config.get("hyper_parameters.pooling_width", (2, 2, 2, 2))
        stride = config.get("hyper_parameters.stride", (1,1))
        dnn_hidden_units = config.get("hyper_parameters.dnn_hidden_units", (128,))
        dnn_dropout = config.get("hyper_parameters.dnn_dropout", 0.0)
        fgcnn_model = net.FGCNN(sparse_input_slot, sparse_feature_size,
                                feature_name, feature_dim,dense_input_slot,
                                conv_kernel_width, conv_filters, new_maps,
                                pooling_width, stride, dnn_hidden_units, dnn_dropout)

        return fgcnn_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        # print(len(batch_data))
        inputs = paddle.to_tensor(np.array(batch_data[0]).astype('int64'))
        inputs = batch_data[0]
        label = batch_data[1]
        return label, inputs
        

    # define loss function by predicts and label
    def create_loss(self, y_pred, label):
        loss = nn.functional.log_loss(
            y_pred, label=paddle.cast(
                label, dtype="float32"))
        avg_cost = paddle.mean(x=loss)
        return avg_cost

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 1e-3)
        optimizer = paddle.optimizer.Adam(
            parameters=dy_model.parameters(),
            learning_rate=lr)
        return optimizer

    def create_metrics(self):
        metrics_list_name = ["auc"]
        auc_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        # 稠密向量
        label, inputs = self.create_feeds(batch_data, config)
        pred = dy_model.forward(inputs)
        loss = self.create_loss(pred, label)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        # print_dict format :{'loss': loss}
        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        # label, sparse_tensor = self.create_feeds(batch_data, config)
        label, inputs = self.create_feeds(batch_data, config)
        pred = dy_model.forward(inputs)
        # pred = dy_model.forward(sparse_tensor)
        loss = self.create_loss(pred, label)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        # print_dict format :{'loss': loss}
        print_dict = {'loss': loss}
        return metrics_list, print_dict