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


class ESMMLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim, num_field,
                 ctr_layer_sizes, cvr_layer_sizes):
        super(ESMMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.num_field = num_field
        self.ctr_layer_sizes = ctr_layer_sizes
        self.cvr_layer_sizes = cvr_layer_sizes

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        # ctr part
        ctr_sizes = [sparse_feature_dim * num_field
                     ] + self.ctr_layer_sizes + [2]
        acts = ["relu" for _ in range(len(self.ctr_layer_sizes))] + [None]
        self._ctr_mlp_layers = []
        for i in range(len(ctr_layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=ctr_sizes[i],
                out_features=ctr_sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(ctr_sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._ctr_mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._ctr_mlp_layers.append(act)

        # ctr part
        cvr_sizes = [sparse_feature_dim * num_field
                     ] + self.cvr_layer_sizes + [2]
        acts = ["relu" for _ in range(len(self.cvr_layer_sizes))] + [None]
        self._cvr_mlp_layers = []
        for i in range(len(cvr_layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=cvr_sizes[i],
                out_features=cvr_sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(cvr_sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._cvr_mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._cvr_mlp_layers.append(act)

    def forward(self, inputs):
        emb = []
        # input feature data
        for data in inputs:
            feat_emb = self.embedding(data)
            # sum pooling
            feat_emb = paddle.sum(feat_emb, axis=1)
            emb.append(feat_emb)
        concat_emb = paddle.concat(x=emb, axis=1)

        ctr_output = concat_emb
        for n_layer in self._ctr_mlp_layers:
            ctr_output = n_layer(ctr_output)

        ctr_out = F.softmax(ctr_output)

        cvr_output = concat_emb
        for n_layer in self._cvr_mlp_layers:
            cvr_output = n_layer(cvr_output)

        cvr_out = F.softmax(cvr_output)

        ctr_prop_one = paddle.slice(ctr_out, axes=[1], starts=[1], ends=[2])
        cvr_prop_one = paddle.slice(cvr_out, axes=[1], starts=[1], ends=[2])
        ctcvr_prop_one = paddle.multiply(x=ctr_prop_one, y=cvr_prop_one)
        ctcvr_prop = paddle.concat(
            x=[1 - ctcvr_prop_one, ctcvr_prop_one], axis=1)

        return ctr_out, ctr_prop_one, cvr_out, cvr_prop_one, ctcvr_prop, ctcvr_prop_one
