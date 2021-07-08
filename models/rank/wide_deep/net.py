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


class WideDeepLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes):
        super(WideDeepLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes

        self.wide_part = paddle.nn.Linear(
            in_features=self.dense_feature_dim,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0, std=1.0 / math.sqrt(self.dense_feature_dim))))

        self.w = self.wide_part.weight

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.v = self.embedding.weight

        self.delta = paddle.create_parameter( 
            shape=[1, self.sparse_feature_number],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value = 0.1))

        sizes = [sparse_feature_dim * num_field + dense_feature_dim
                 ] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

    def forward(self, sparse_inputs, dense_inputs):
        # wide part
        wide_output = self.wide_part(dense_inputs)

        t = 0.1
        u = paddle.uniform(min = 0, max = 1, shape = self.delta.shape)
        out = (paddle.log(self.delta) - paddle.log(1 - self.delta) + paddle.log(u) - paddle.log(1 - u)) / t
        z = paddle.nn.functional.sigmoid(out)
        print("z: {}".format(z))
        theta = 0.4 * paddle.ones(paddle.shape(z), dtype='float32') # assume that each feature is selected with prob. 0.4
        alpha = paddle.log(1 - theta) - paddle.log(theta)
        print("alpha: {}".format(alpha))
        factors = paddle.multiply(z, alpha)

        # deep part
        sparse_embs = []
        for s_input in sparse_inputs:  # s_input: 50,1(tensor)
            emb = self.embedding(s_input)  # emb: 50,1,9
            batch_size = len(sparse_inputs)
            for i in range(batch_size):
                select_factor = factors[0][s_input[i, 0]]
                emb = emb * select_factor
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            sparse_embs.append(emb)  # emb: 50,9   sparse_embs: 13
        # 不同样本中的13个特征id是不同的
        deep_output = paddle.concat(x = sparse_embs + [dense_inputs], axis=1)
        for n_layer in self._mlp_layers:
            deep_output = n_layer(deep_output)

        prediction = paddle.add(x = wide_output, y = deep_output)
        pred = F.sigmoid(prediction)
        return pred, self.w, z, self.v, alpha 
