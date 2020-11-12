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
from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


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
        sentence_left = paddle.fluid.data(
            name="sentence_left",
            shape=[-1, self.sentence_left_size, 1],
            dtype='int64',
            lod_level=0)
        sentence_right = paddle.fluid.data(
            name="sentence_right",
            shape=[-1, self.sentence_right_size, 1],
            dtype='int64',
            lod_level=0)
        return [sentence_left, sentence_right]

    def embedding_layer(self, input):
        """
        embedding layer
        """
        if os.path.isfile(self.emb_path):
            embedding_array = np.load(self.emb_path)
            emb = paddle.static.nn.embedding(
                input=input,
                size=[self.vocab_size, self.emb_size],
                padding_idx=0,
                param_attr=paddle.ParamAttr(
                    name="word_embedding",
                    initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                        embedding_array)))
        else:
            emb = paddle.static.nn.embedding(
                input=input,
                size=[self.vocab_size, self.emb_size],
                padding_idx=0,
                param_attr=paddle.ParamAttr(
                    name="word_embedding",
                    initializer=paddle.fluid.initializer.Xavier()))

        return emb

    def conv_pool_layer(self, input):
        """
        convolution and pool layer
        """
        # data format NCHW
        # same padding
        conv = paddle.fluid.layers.nn.conv2d(
            input=input,
            num_filters=self.kernel_num,
            stride=1,
            padding="SAME",
            filter_size=self.conv_filter,
            act=self.conv_act)
        pool = paddle.fluid.layers.pool2d(
            input=conv,
            pool_size=self.pool_size,
            pool_stride=self.pool_stride,
            pool_type=self.pool_type,
            pool_padding=self.pool_padding)
        return pool

    def net(self, inputs, is_infer=False):
        left_emb = self.embedding_layer(inputs[0])
        right_emb = self.embedding_layer(inputs[1])
        cross = paddle.fluid.layers.matmul(
            left_emb, right_emb, transpose_y=True)
        cross = paddle.fluid.layers.reshape(
            cross, [-1, 1, cross.shape[1], cross.shape[2]])
        conv_pool = self.conv_pool_layer(input=cross)
        relu_hid = paddle.static.nn.fc(x=conv_pool,
                                       size=self.hidden_size,
                                       activation=self.hidden_act)
        prediction = paddle.static.nn.fc(
            x=relu_hid,
            size=self.out_size, )

        if is_infer:
            self._infer_results["prediction"] = prediction
            return

        pos = paddle.slice(
            prediction, axes=[0, 1], starts=[0, 0], ends=[64, 1])
        neg = paddle.slice(
            prediction, axes=[0, 1], starts=[64, 0], ends=[128, 1])
        loss_part1 = paddle.fluid.layers.nn.elementwise_sub(
            paddle.full(
                shape=[64, 1], fill_value=1.0), pos)
        loss_part2 = paddle.add(x=loss_part1, y=neg)
        loss_part3 = paddle.maximum(
            x=paddle.full(
                shape=[64, 1], fill_value=0.0), y=loss_part2)

        avg_cost = paddle.mean(x=loss_part3)
        self._cost = avg_cost
