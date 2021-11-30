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
from net import ENSFMLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.num_users = self.config.get("hyper_parameters.num_users")
        self.num_items = self.config.get("hyper_parameters.num_items")
        self.embedding_size = self.config.get("hyper_parameters.mf_dim")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")

    def create_feeds(self, is_infer=False):
        user_input = paddle.static.data(
            name="user_input", shape=[-1, 1], dtype='int64')
        item_attribute = paddle.static.data(
            name="item_attribute", shape=[-1, 1], dtype='int64')
        if is_infer:
            feeds_list = [user_input, item_attribute]
        else:
            item_input = paddle.static.data(
                name="item_input", shape=[-1, 1], dtype='int64')
            item_bind_M = paddle.static.data(
                name="item_bind_M", shape=[1], dtype='int64')
            feeds_list = [user_input, item_attribute, item_input, item_bind_M]
        return feeds_list

    def net(self, input, is_infer=False):
        model = ENSFMLayer(self.num_users, self.num_items, self.embedding_size)
        prediction = model(*input)
        self.inference_target_var = prediction
        if is_infer:
            fetch_dict = {
                "user": input[0],
                'prediction': prediction,
            }
            return fetch_dict

        pre, pos_r, q_emb, p_emb, H_i_emb = prediction
        weight = self.config.get('hyper_parameters.negative_weight', 0.5)
        loss = weight * paddle.sum(
            paddle.sum(paddle.sum(q_emb.unsqueeze(-1) * q_emb.unsqueeze(-2), 0)
                       * paddle.sum(p_emb.unsqueeze(-1) * p_emb.unsqueeze(-2), 0)
                       * paddle.matmul(H_i_emb, H_i_emb, transpose_y=True), 0), 0)
        loss += paddle.sum((1.0 - weight) * paddle.square(pos_r) - 2.0 * pos_r)
        # print(avg_cost)
        self._cost = loss
        fetch_dict = {'Loss': loss}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adagrad(
            learning_rate=self.learning_rate)
        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)
