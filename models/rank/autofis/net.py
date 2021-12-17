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
from itertools import combinations


def generate_pairs(ranges, mask=None, order=2):
    res = []
    for i in range(order):
        res.append([])
    for i, pair in enumerate(list(combinations(ranges, order))):
        if mask is None or mask[i] == 1:
            for j in range(order):
                res[j].append(pair[j])
    return paddle.to_tensor(res[0]), paddle.to_tensor(res[1])


class AutoDeepFMLayer(nn.Layer):
    def __init__(self, num_inputs, input_size, embedding_size, width, depth, pairs, stage):
        super().__init__()
        self.stage = stage
        self.depth = depth
        self.w_embeddings = nn.Embedding(input_size, 1)
        self.v_embeddings = nn.Embedding(input_size, embedding_size)
        in_features = [num_inputs * embedding_size] + [width] * depth
        out_features = [width] * depth + [1]
        self.bn = nn.LayerList([nn.BatchNorm1D(width) for _ in range(width)])
        self.linear = nn.LayerList([nn.Linear(*_) for _ in zip(in_features, out_features)])

        self.mask = paddle.create_parameter([1, pairs], 'float32',
                                            default_initializer=nn.initializer.Uniform(0.6 - 0.001, 0.6 + 0.001))
        # self.bn2 = nn.BatchNorm1D(pairs)
        self.bn2 = nn.Identity()

        self.comb_mask = None if stage == 0 else np.load('comb_mask.npy')

    def forward(self, inputs):
        # 对应embedding lookup
        xw = self.w_embeddings(inputs).squeeze(-1)
        xv = self.v_embeddings(inputs)
        # 对应liner
        h = xv.flatten(1)
        # 对应bin_mlp
        for i in range(self.depth + 1):
            h = self.linear[i](h)
            # h = self.bn[i](h)
            if i != self.depth:
                h = F.relu(h)
        h = h.squeeze(-1)
        l = xw.sum(1)
        # 对应generate_pairs
        cols, rows = generate_pairs(range(xv.shape[1]), mask=self.comb_mask)
        cols = paddle.to_tensor(cols).unsqueeze(-1)
        rows = paddle.to_tensor(rows).unsqueeze(-1)
        left = paddle.gather(xv, cols, 1)
        right = paddle.gather(xv, rows, 1)
        level_2_matrix = (left * right).sum(-1)
        level_2_matrix = self.bn2(level_2_matrix) * self.mask
        fm_out = level_2_matrix.sum(-1)
        # print(l.shape, fm_out.shape, h.shape)
        return F.sigmoid(l + fm_out + h)
