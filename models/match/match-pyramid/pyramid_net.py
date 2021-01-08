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
import numpy as np
import os


class MatchPyramidLayer(nn.Layer):
    def __init__(self, emb_path, vocab_size, emb_size, kernel_num, conv_filter,
                 conv_act, hidden_size, out_size, pool_size, pool_stride,
                 pool_padding, pool_type, hidden_act):
        super(MatchPyramidLayer, self).__init__()
        self.emb_path = emb_path
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.kernel_num = kernel_num
        self.conv_filter = conv_filter
        self.conv_act = conv_act
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.pool_type = pool_type
        self.hidden_act = hidden_act

        if os.path.isfile(self.emb_path):
            embedding_array = np.load(self.emb_path)
            self.emb = paddle.nn.Embedding(
                self.vocab_size,
                self.emb_size,
                padding_idx=193367,
                weight_attr=paddle.ParamAttr(
                    name="word_embedding",
                    initializer=nn.initializer.Assign(embedding_array)))
        else:
            self.emb = paddle.nn.Embedding(
                self.vocab_size,
                self.emb_size,
                padding_idx=193367,
                weight_attr=paddle.ParamAttr(
                    name="word_embedding",
                    initializer=nn.initializer.XavierNormal()))

        self.conv = nn.Conv2D(
            in_channels=1,
            out_channels=self.kernel_num,
            stride=1,
            padding="SAME",
            kernel_size=self.conv_filter)
        self.fc1 = paddle.nn.Linear(
            in_features=240, out_features=self.hidden_size)
        self.fc2 = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.out_size)

    def forward(self, inputs):
        left_emb = self.emb(inputs[0])
        right_emb = self.emb(inputs[1])
        cross = paddle.matmul(left_emb, right_emb, transpose_y=True)
        cross = paddle.reshape(cross, [-1, 1, cross.shape[1], cross.shape[2]])
        conv = self.conv(cross)
        if self.conv_act == "relu":
            conv = F.relu(conv)
        if self.pool_type == "max":
            pool = F.max_pool2d(
                conv,
                kernel_size=self.pool_size,
                stride=self.pool_stride,
                padding=self.pool_padding)
        reshape = paddle.reshape(pool, [
            -1, list(pool.shape)[1] * list(pool.shape)[2] * list(pool.shape)[3]
        ])
        hid = self.fc1(reshape)
        if self.hidden_act == "relu":
            relu_hid = F.relu(hid)
        prediction = self.fc2(relu_hid)
        return prediction
