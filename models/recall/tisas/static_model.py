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

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from net import TiSASRecLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.num_users = self.config.get("hyper_parameters.num_users")
        self.num_items = self.config.get("hyper_parameters.num_items")
        self.hidden_units = self.config.get("hyper_parameters.hidden_units")
        self.maxlen = self.config.get("hyper_parameters.maxlen")
        self.time_span = self.config.get("hyper_parameters.time_span")
        self.num_blocks = self.config.get("hyper_parameters.num_blocks")
        self.num_heads = self.config.get("hyper_parameters.num_heads")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")

    def create_feeds(self, is_infer=False):

        log_seqs = paddle.static.data(
            name="log_seqs", shape=[-1, 50], dtype='int64')
        time_matrices = paddle.static.data(
            name="time_matrices", shape=[-1, 50, 50], dtype='int64')
        item_indices = paddle.static.data(
            name="item_indices", shape=[-1, 50, 50], dtype='int64')
        pos_seqs = paddle.static.data(
            name="pos_seqs", shape=[-1, 50], dtype='int64')
        neg_seqs = paddle.static.data(
            name="neg_seqs", shape=[-1, 50], dtype='int64')
        if is_infer:
            return log_seqs, time_matrices, item_indices
        return log_seqs, time_matrices, pos_seqs, neg_seqs

    def net(self, input, is_infer=False):
        model = TiSASRecLayer(self.num_users, self.num_items,
                              self.hidden_units, self.maxlen, self.time_span,
                              self.num_blocks, self.num_heads)

        if is_infer:
            prediction = model(*input)
        else:
            prediction = model(
                input[0], input[1], pos_seqs=input[2], neg_seqs=input[3])

        self.inference_target_var = prediction
        if is_infer:
            fetch_dict = {
                "user": input[0],
                'prediction': prediction,
            }
            return fetch_dict

        mask = input[3] != 0
        avg_cost = self.create_loss(prediction, mask)
        # print(avg_cost)
        self._cost = avg_cost
        fetch_dict = {'Loss': avg_cost}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adam(
            learning_rate=self.learning_rate, lazy_mode=True)
        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)

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
