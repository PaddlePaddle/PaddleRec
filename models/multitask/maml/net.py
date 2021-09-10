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
import copy
import numpy as np


class MAMLLayer(paddle.nn.Layer):
    def __init__(self, conv_stride, conv_padding, conv_kernal, bn_channel):
        super(MAMLLayer, self).__init__()
        # ------------------------第1个conv2d-------------------------
        self.conv_1 = paddle.nn.Conv2D(
            1,
            64,
            conv_kernal,
            stride=conv_stride,
            padding=conv_padding,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.KaimingNormal()),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        # ------------------------第1个BatchNorm-------------------------
        self.BN_1 = paddle.nn.BatchNorm2D(
            bn_channel,
            momentum=0.9,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1)),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        # ------------------------第2个conv2d------------------------
        self.conv_2 = paddle.nn.Conv2D(
            64,
            64,
            conv_kernal,
            stride=conv_stride,
            padding=conv_padding,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.KaimingNormal()),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        # ------------------------第2个BatchNorm-------------------------
        self.BN_2 = paddle.nn.BatchNorm2D(
            bn_channel,
            momentum=0.9,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1)),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        # ------------------------第3个conv2d------------------------
        self.conv_3 = paddle.nn.Conv2D(
            64,
            64,
            conv_kernal,
            stride=conv_stride,
            padding=conv_padding,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.KaimingNormal()),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        # ------------------------第3个BatchNorm-------------------------
        self.BN_3 = paddle.nn.BatchNorm2D(
            bn_channel,
            momentum=0.9,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1)),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        # ------------------------第4个conv2d------------------------
        self.conv_4 = paddle.nn.Conv2D(
            64,
            64,
            conv_kernal,
            stride=conv_stride,
            padding=conv_padding,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.KaimingNormal()),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        # ------------------------第4个BatchNorm-------------------------
        self.BN_4 = paddle.nn.BatchNorm2D(
            bn_channel,
            momentum=0.9,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1)),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        # ------------------------全连接层------------------------
        self.linear = paddle.nn.Linear(
            in_features=64,
            out_features=5,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal()),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))

    def forward(self, x):

        x = self.conv_1(x)
        x = self.BN_1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)  # 第1个MAX_POOL层

        x = self.conv_2(x)
        x = self.BN_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)  # 第2个MAX_POOL层

        x = self.conv_3(x)
        x = self.BN_3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)  # 第3个MAX_POOL层

        x = self.conv_4(x)
        x = self.BN_4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)  # 第4个MAX_POOL层

        x = paddle.flatten(x, 1, -1)  ## flatten
        x = self.linear(x)  # linear

        output = x

        return output
