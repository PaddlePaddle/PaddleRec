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
import numpy as np


class Tower(nn.Layer):
    def __init__(self,
                 input_dim: int,
                 dims=[128, 64, 32],
                 drop_prob=[0.1, 0.3, 0.3]):
        super(Tower, self).__init__()
        self.dims = dims
        self.drop_prob = drop_prob
        self.layer = nn.Sequential(
            nn.Linear(input_dim, dims[0]),
            nn.ReLU(),
            nn.Dropout(drop_prob[0]),
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Dropout(drop_prob[1]),
            nn.Linear(dims[1], dims[2]), nn.ReLU(), nn.Dropout(drop_prob[2]))

    def forward(self, x):
        x = paddle.flatten(x, 1)
        x = self.layer(x)
        return x


class Attention(nn.Layer):
    """Self-attention layer for click and purchase"""

    def __init__(self, dim=32):
        super(Attention, self).__init__()
        self.dim = dim
        self.q_layer = nn.Linear(dim, dim, bias_attr=False)
        self.k_layer = nn.Linear(dim, dim, bias_attr=False)
        self.v_layer = nn.Linear(dim, dim, bias_attr=False)
        self.softmax = nn.Softmax(1)

    def forward(self, inputs):
        Q = self.q_layer(inputs)
        K = self.k_layer(inputs)
        V = self.v_layer(inputs)
        a = (Q * K).sum(-1) / (self.dim**0.5)
        a = self.softmax(a)
        outputs = (a.unsqueeze(-1) * V).sum(1)
        return outputs


class AITM(nn.Layer):
    def __init__(self,
                 feature_vocabulary,
                 embedding_size,
                 tower_dims=[128, 64, 32],
                 drop_prob=[0.1, 0.3, 0.3]):
        super(AITM, self).__init__()
        self.feature_vocabulary = feature_vocabulary
        self.feature_names = sorted(list(feature_vocabulary.keys()))
        self.embedding_size = embedding_size
        self.embedding_dict = nn.LayerList()
        self.__init_weight()

        self.tower_input_size = len(feature_vocabulary) * embedding_size
        self.click_tower = Tower(self.tower_input_size, tower_dims, drop_prob)
        self.conversion_tower = Tower(self.tower_input_size, tower_dims,
                                      drop_prob)
        self.attention_layer = Attention(tower_dims[-1])

        self.info_layer = nn.Sequential(
            nn.Linear(tower_dims[-1], 32),
            nn.ReLU(), nn.Dropout(drop_prob[-1]))

        self.click_layer = nn.Sequential(
            nn.Linear(tower_dims[-1], 1), nn.Sigmoid())
        self.conversion_layer = nn.Sequential(
            nn.Linear(tower_dims[-1], 1), nn.Sigmoid())

    def __init_weight(self, ):
        for name, size in self.feature_vocabulary.items():
            emb = nn.Embedding(size, self.embedding_size)
            self.embedding_dict.append(emb)

    def forward(self, x):
        feature_embedding = []
        for i in range(len(x)):
            embed = self.embedding_dict[i](x[i])
            feature_embedding.append(embed)

        feature_embedding = paddle.concat(feature_embedding, 1)
        tower_click = self.click_tower(feature_embedding)

        tower_conversion = paddle.unsqueeze(
            self.conversion_tower(feature_embedding), 1)

        info = paddle.unsqueeze(self.info_layer(tower_click), 1)

        ait = self.attention_layer(paddle.concat([tower_conversion, info], 1))

        click = paddle.squeeze(self.click_layer(tower_click), 1)
        conversion = paddle.squeeze(self.conversion_layer(ait), 1)

        return click, conversion
