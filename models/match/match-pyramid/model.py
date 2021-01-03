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

import os
import sys
import random
import numpy as np
import paddle
import paddle.fluid as fluid
from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase
from pyramid_net import MatchPyramidLayer


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.emb_path = envs.get_global_env("hyper_parameters.emb_path")
        self.sentence_left_size = envs.get_global_env(
            "hyper_parameters.sentence_left_size")
        self.sentence_right_size = envs.get_global_env(
            "hyper_parameters.sentence_right_size")
        self.vocab_size = envs.get_global_env("hyper_parameters.vocab_size")
        self.emb_size = envs.get_global_env("hyper_parameters.emb_size")
        self.kernel_num = envs.get_global_env("hyper_parameters.kernel_num")
        self.hidden_size = envs.get_global_env("hyper_parameters.hidden_size")
        self.hidden_act = envs.get_global_env("hyper_parameters.hidden_act")
        self.out_size = envs.get_global_env("hyper_parameters.out_size")
        self.channels = envs.get_global_env("hyper_parameters.channels")
        self.conv_filter = envs.get_global_env("hyper_parameters.conv_filter")
        self.conv_act = envs.get_global_env("hyper_parameters.conv_act")
        self.pool_size = envs.get_global_env("hyper_parameters.pool_size")
        self.pool_stride = envs.get_global_env("hyper_parameters.pool_stride")
        self.pool_type = envs.get_global_env("hyper_parameters.pool_type")
        self.pool_padding = envs.get_global_env(
            "hyper_parameters.pool_padding")

    def input_data(self, is_infer=False, **kwargs):
        sentence_left = paddle.static.data(
            name="sentence_left",
            shape=[-1, self.sentence_left_size],
            dtype='int64',
            lod_level=0)
        sentence_right = paddle.static.data(
            name="sentence_right",
            shape=[-1, self.sentence_right_size],
            dtype='int64',
            lod_level=0)
        return [sentence_left, sentence_right]

    def net(self, inputs, is_infer=False):
        pyramid_model = MatchPyramidLayer(
            self.emb_path, self.vocab_size, self.emb_size, self.kernel_num,
            self.conv_filter, self.conv_act, self.hidden_size, self.out_size,
            self.pool_size, self.pool_stride, self.pool_padding,
            self.pool_type, self.hidden_act)
        prediction = pyramid_model(inputs)

        if is_infer:
            self._infer_results["prediction"] = prediction
            return

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
        self._cost = avg_cost
