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
        sentence_left = fluid.data(
            name="sentence_left",
            shape=[-1, self.sentence_left_size, 1],
            dtype='int64',
            lod_level=0)
        sentence_right = fluid.data(
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
            emb = fluid.layers.embedding(
                input=input,
                size=[self.vocab_size, self.emb_size],
                padding_idx=0,
                param_attr=fluid.ParamAttr(
                    name="word_embedding",
                    initializer=fluid.initializer.NumpyArrayInitializer(
                        embedding_array)))
        else:
            emb = fluid.layers.embedding(
                input=input,
                size=[self.vocab_size, self.emb_size],
                padding_idx=0,
                param_attr=fluid.ParamAttr(
                    name="word_embedding",
                    initializer=fluid.initializer.Xavier()))

        return emb

    def conv_pool_layer(self, input):
        """
        convolution and pool layer
        """
        # data format NCHW
        # same padding
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=self.kernel_num,
            stride=1,
            padding="SAME",
            filter_size=self.conv_filter,
            act=self.conv_act)
        pool = fluid.layers.pool2d(
            input=conv,
            pool_size=self.pool_size,
            pool_stride=self.pool_stride,
            pool_type=self.pool_type,
            pool_padding=self.pool_padding)
        return pool

    def net(self, inputs, is_infer=False):
        left_emb = self.embedding_layer(inputs[0])
        right_emb = self.embedding_layer(inputs[1])
        cross = fluid.layers.matmul(left_emb, right_emb, transpose_y=True)
        cross = fluid.layers.reshape(cross,
                                     [-1, 1, cross.shape[1], cross.shape[2]])
        conv_pool = self.conv_pool_layer(input=cross)
        relu_hid = fluid.layers.fc(input=conv_pool,
                                   size=self.hidden_size,
                                   act=self.hidden_act)
        prediction = fluid.layers.fc(
            input=relu_hid,
            size=self.out_size, )

        if is_infer:
            self._infer_results["prediction"] = prediction
            return

        pos = fluid.layers.slice(
            prediction, axes=[0, 1], starts=[0, 0], ends=[64, 1])
        neg = fluid.layers.slice(
            prediction, axes=[0, 1], starts=[64, 0], ends=[128, 1])
        loss_part1 = fluid.layers.elementwise_sub(
            fluid.layers.fill_constant(
                shape=[64, 1], value=1.0, dtype='float32'),
            pos)
        loss_part2 = fluid.layers.elementwise_add(loss_part1, neg)
        loss_part3 = fluid.layers.elementwise_max(
            fluid.layers.fill_constant(
                shape=[64, 1], value=0.0, dtype='float32'),
            loss_part2)

        avg_cost = fluid.layers.mean(loss_part3)
        self._cost = avg_cost
