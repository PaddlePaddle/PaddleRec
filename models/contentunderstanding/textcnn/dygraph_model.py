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
        dict_dim = config.get("hyper_parameters.dict_dim")
        max_len = config.get("hyper_parameters.max_len")
        cnn_dim = config.get("hyper_parameters.cnn_dim")
        cnn_filter_size1 = config.get("hyper_parameters.cnn_filter_size1")
        cnn_filter_size2 = config.get("hyper_parameters.cnn_filter_size2")
        cnn_filter_size3 = config.get("hyper_parameters.cnn_filter_size3")
        filter_sizes = [cnn_filter_size1, cnn_filter_size2, cnn_filter_size3]
        emb_dim = config.get("hyper_parameters.emb_dim")
        hid_dim = config.get("hyper_parameters.hid_dim")
        class_dim = config.get("hyper_parameters.class_dim")
        is_sparse = config.get("hyper_parameters.is_sparse")

        textcnn_model = net.TextCNNLayer(
            dict_dim,
            emb_dim,
            class_dim,
            cnn_dim=cnn_dim,
            filter_sizes=filter_sizes,
            hidden_size=hid_dim)

        return textcnn_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        input_data = paddle.to_tensor(batch_data[0].numpy().astype('int64')
                                      .reshape(-1, 100))
        label = paddle.to_tensor(batch_data[1].numpy().astype('int64').reshape(
            -1, 1))

        return input_data, label

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        cost = paddle.nn.functional.cross_entropy(input=pred, label=label)
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
        metrics_list_name = ["acc"]
        acc_metric = paddle.metric.Accuracy()
        metrics_list = [acc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        input_data, label = self.create_feeds(batch_data, config)
        pred = dy_model.forward(input_data)
        loss = self.create_loss(pred, label)

        # update metrics
        prediction = paddle.nn.functional.softmax(pred)
        correct = metrics_list[0].compute(prediction, label)
        metrics_list[0].update(correct)

        print_dict = {'loss': loss}
        # print_dict = None
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        input_data, label = self.create_feeds(batch_data, config)
        pred = dy_model.forward(input_data)

        # update metrics
        prediction = paddle.nn.functional.softmax(pred)
        correct = metrics_list[0].compute(prediction, label)
        metrics_list[0].update(correct)
        return metrics_list, None
