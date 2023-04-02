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
        num_users = config.get("hyper_parameters.num_users")
        num_items = config.get("hyper_parameters.num_items")
        hidden_units = config.get("hyper_parameters.hidden_units")
        maxlen = config.get("hyper_parameters.maxlen")
        time_span = config.get("hyper_parameters.time_span")
        num_blocks = config.get("hyper_parameters.num_blocks")
        num_heads = config.get("hyper_parameters.num_heads")
        enc_fm_model = net.TiSASRecLayer(num_users, num_items, hidden_units,
                                         maxlen, time_span, num_blocks,
                                         num_heads)
        return enc_fm_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data):
        return batch_data

    # define loss function by predicts and label
    def create_loss(self, prediction, mask):
        loss_fct = paddle.nn.BCEWithLogitsLoss()
        pos_logits, neg_logits = prediction
        pos_labels, neg_labels = paddle.ones_like(
            pos_logits), paddle.zeros_like(neg_logits)
        loss = loss_fct(
            paddle.masked_select(pos_logits, mask),
            paddle.masked_select(pos_labels, mask))
        loss += loss_fct(
            paddle.masked_select(neg_logits, mask),
            paddle.masked_select(neg_labels, mask))
        return loss

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(
            parameters=dy_model.parameters(),
            learning_rate=lr,
            beta1=0.9,
            beta2=0.98)
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric

    def create_metrics(self):
        metrics_list_name = []
        metrics_list = []
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        inputs = self.create_feeds(batch_data)
        seq, time_matrix, pos, neg = inputs
        prediction = dy_model.forward(
            seq, time_matrix, pos_seqs=pos, neg_seqs=neg)
        mask = pos != 0
        loss = self.create_loss(prediction, mask)
        # update metrics
        print_dict = {"loss": loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        inputs = self.create_feeds(batch_data)

        prediction = dy_model.forward(*inputs)
        # update metrics
        print_dict = {
            "user": inputs[0],
            "prediction": prediction,
        }
        return metrics_list, print_dict
