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
import paddle.nn.functional as F
import net


class DygraphModel():
    # define model
    def create_model(self, config):
        num_users = config.get("hyper_parameters.num_users")
        num_items = config.get("hyper_parameters.num_items")
        num_bizs = config.get("hyper_parameters.num_bizs")
        hidden_units = config.get("hyper_parameters.hidden_units")
        f_max_len = config.get("hyper_parameters.f_max_len")
        k = config.get("hyper_parameters.k")
        u_max_i = config.get("hyper_parameters.u_max_i")
        u_max_f = config.get("hyper_parameters.u_max_f")
        u_max_pack = config.get("hyper_parameters.u_max_pack")
        pack_max_nei_b = config.get("hyper_parameters.pack_max_nei_b")
        pack_max_nei_f = config.get("hyper_parameters.pack_max_nei_f")
        dropout_rate = config.get("hyper_parameters.dropout_rate")

        enc_fm_model = net.IPRECLayer(
            num_users,
            num_items,
            num_bizs,
            hidden_units,
            f_max_len,
            k,
            u_max_i,
            u_max_f,
            u_max_pack,
            pack_max_nei_b,
            pack_max_nei_f,
            dropout_rate)
        return enc_fm_model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, batch_data):
        return batch_data

    # define loss function by predicts and label
    def create_loss(self, prediction, label):
        loss = F.binary_cross_entropy(prediction, label)
        return loss

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.0001)
        optimizer = paddle.optimizer.Adam(
            parameters=dy_model.parameters(),
            learning_rate=lr,
            weight_decay=1e-5,
        )
        return optimizer

    def create_metrics(self):
        metrics_list_name = ["auc"]
        auc_metric = paddle.metric.Auc("ROC", num_thresholds=1000000)
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        inputs = self.create_feeds(batch_data)
        *inputs, label = inputs
        label = label.astype('float32')
        prediction = dy_model.forward(*inputs)
        loss = self.create_loss(prediction, label)
        predict_2d = paddle.concat([1 - prediction, prediction], axis=1)
        metrics_list[0].update(preds=predict_2d, labels=label)
        # update metrics
        print_dict = {"loss": loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        inputs = self.create_feeds(batch_data)
        *inputs, label = inputs
        prediction = dy_model.forward(*inputs)
        # update metrics
        predict_2d = paddle.concat(x=[1 - prediction, prediction], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        return metrics_list, None

    def forward(self, dy_model, batch_data, config):
        inputs = self.create_feeds(batch_data)
        *inputs, label = inputs
        prediction = dy_model.forward(*inputs)
        return prediction, label