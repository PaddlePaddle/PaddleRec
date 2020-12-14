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


class TextCNNLayer(nn.Layer):
    def __init__(self,
                 dict_dim,
                 emb_dim,
                 class_dim,
                 padding_idx=0,
                 cnn_dim=128,
                 filter_sizes=[1, 2, 3],
                 hidden_size=96,
                 conv_layer_activation=nn.Tanh()):
        super(TextCNNLayer, self).__init__()
        self.dict_dim = dict_dim
        self.emb_dim = emb_dim
        self.class_dim = class_dim
        self.padding_idx = padding_idx
        self.cnn_dim = cnn_dim
        self.filter_sizes = filter_sizes
        self.hidden_size = hidden_size
        self.conv_layer_activation = conv_layer_activation

        self.embedding = paddle.nn.Embedding(
            self.dict_dim, self.emb_dim, padding_idx=self.padding_idx)

        self.convs = [
            nn.Conv2D(
                in_channels=1,
                out_channels=self.cnn_dim,
                kernel_size=(i, self.emb_dim)) for i in self.filter_sizes
        ]

        self.projection_layer = paddle.nn.Linear(
            in_features=self.cnn_dim * len(self.filter_sizes),
            out_features=self.hidden_size)

        self.output_layer = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.class_dim)

    def forward(self, inputs):
        emb = self.embedding(inputs)
        emb = emb.unsqueeze(1)
        # convolution layer
        convs_out = [
            self.conv_layer_activation(conv(emb)).squeeze(3)
            for conv in self.convs
        ]
        # pool layer
        maxpool_out = [
            F.max_pool1d(
                t, kernel_size=t.shape[2]).squeeze(2) for t in convs_out
        ]

        conv_pool_out = paddle.concat(maxpool_out, axis=1)
        conv_pool_out = self.projection_layer(conv_pool_out)

        act_out = paddle.tanh(conv_pool_out)

        logits = self.output_layer(act_out)
        return logits
