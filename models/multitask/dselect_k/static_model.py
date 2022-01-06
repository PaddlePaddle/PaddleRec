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

from net import MMoELayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.feature_size = self.config.get('hyper_parameters.feature_size',
                                            None)
        self.expert_num = self.config.get('hyper_parameters.expert_num', None)
        self.expert_size = self.config.get('hyper_parameters.expert_size',
                                           None)
        self.tower_size = self.config.get('hyper_parameters.tower_size', None)
        self.gate_num = self.config.get('hyper_parameters.gate_num', None)
        self.top_k = self.config.get('hyper_parameters.top_k', 2)

        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")

    def create_feeds(self, is_infer=False):
        inputs = paddle.static.data(
            name="input", shape=[-1, 1, 36, 36], dtype="float32")
        label_left = paddle.static.data(
            name="label_left", shape=[-1, 1], dtype="int64")
        label_right = paddle.static.data(
            name="label_right", shape=[-1, 1], dtype="int64")

        if is_infer:
            return [inputs, label_left, label_right]
        else:
            return [inputs, label_left, label_right]

    def net(self, inputs, is_infer=False):
        input_data = inputs[0]
        label_left = paddle.reshape(inputs[1], [-1, 1])
        label_right = paddle.reshape(inputs[2], [-1, 1])

        MMoE = MMoELayer(self.feature_size, self.expert_num, self.expert_size,
                         self.tower_size, self.gate_num, self.top_k)
        pred_left, pred_right = MMoE.forward(input_data)

        acc_left = paddle.static.accuracy(input=pred_left, label=label_left)
        acc_right = paddle.static.accuracy(input=pred_right, label=label_right)

        if is_infer:
            fetch_dict = {'acc_left': acc_left, 'acc_right': acc_right}
            return fetch_dict

        cost_left = paddle.nn.functional.cross_entropy(
            input=pred_left, label=label_left)
        cost_right = paddle.nn.functional.cross_entropy(
            input=pred_right, label=label_right)

        cost = cost_left + cost_right

        self._cost = cost
        fetch_dict = {
            'cost': cost,
            'acc_left': acc_left,
            'acc_right': acc_right
        }
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
