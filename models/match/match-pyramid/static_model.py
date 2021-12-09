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
from net import MatchPyramidLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.emb_path = self.config.get("hyper_parameters.emb_path")
        self.sentence_left_size = self.config.get(
            "hyper_parameters.sentence_left_size")
        self.sentence_right_size = self.config.get(
            "hyper_parameters.sentence_right_size")
        self.vocab_size = self.config.get("hyper_parameters.vocab_size")
        self.emb_size = self.config.get("hyper_parameters.emb_size")
        self.kernel_num = self.config.get("hyper_parameters.kernel_num")
        self.hidden_size = self.config.get("hyper_parameters.hidden_size")
        self.hidden_act = self.config.get("hyper_parameters.hidden_act")
        self.out_size = self.config.get("hyper_parameters.out_size")
        self.channels = self.config.get("hyper_parameters.channels")
        self.conv_filter = self.config.get("hyper_parameters.conv_filter")
        self.conv_act = self.config.get("hyper_parameters.conv_act")
        self.pool_size = self.config.get("hyper_parameters.pool_size")
        self.pool_stride = self.config.get("hyper_parameters.pool_stride")
        self.pool_type = self.config.get("hyper_parameters.pool_type")
        self.pool_padding = self.config.get("hyper_parameters.pool_padding")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")

    def create_feeds(self, is_infer=False):
        sentence_left = paddle.static.data(
            name="sentence_left",
            shape=[-1, self.sentence_left_size],
            dtype='int64')
        sentence_right = paddle.static.data(
            name="sentence_right",
            shape=[-1, self.sentence_right_size],
            dtype='int64')
        feeds_list = [sentence_left, sentence_right]
        return feeds_list

    def net(self, input, is_infer=False):
        pyramid_model = MatchPyramidLayer(
            self.emb_path, self.vocab_size, self.emb_size, self.kernel_num,
            self.conv_filter, self.conv_act, self.hidden_size, self.out_size,
            self.pool_size, self.pool_stride, self.pool_padding,
            self.pool_type, self.hidden_act)
        prediction = pyramid_model.forward(input)

        if is_infer:
            fetch_dict = {'prediction': prediction}
            return fetch_dict

        # calculate hinge loss
        pos = paddle.slice(
            prediction, axes=[0, 1], starts=[0, 0], ends=[64, 1])
        neg = paddle.slice(
            prediction, axes=[0, 1], starts=[64, 0], ends=[128, 1])
        loss_part1 = paddle.subtract(
            paddle.full(
                shape=[64, 1], fill_value=1.0, dtype='float32'), pos)
        loss_part2 = paddle.add(loss_part1, neg)
        loss_part3 = paddle.maximum(
            paddle.full(
                shape=[64, 1], fill_value=0.0, dtype='float32'),
            loss_part2)
        avg_cost = paddle.mean(loss_part3)

        self.inference_target_var = avg_cost
        self._cost = avg_cost

        fetch_dict = {'cost': avg_cost}
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
