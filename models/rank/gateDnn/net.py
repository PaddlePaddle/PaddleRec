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
import math


class GateDNNLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes, use_embedding_gate,
                 use_hidden_gate):
        super(GateDNNLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes
        self.use_embedding_gate = use_embedding_gate
        self.use_hidden_gate = use_hidden_gate
        if self.use_embedding_gate:
            self.embedding_gate_weight = [
                paddle.create_parameter(
                    shape=[1],
                    dtype="float32",
                    name='embedding_gate_weight_%d' % i,
                    default_initializer=paddle.nn.initializer.Normal(std=1.0))
                for i in range(num_field)
            ]
            for i in range(num_field):
                self.add_parameter('embedding_gate_weight_%d' % i,
                                   self.embedding_gate_weight[i])
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))
        sizes = [sparse_feature_dim * num_field + dense_feature_dim
                 ] + self.layer_sizes
        self._mlp_layers = []
        self._hidden_gate_weight = []
        self.last_layer = paddle.nn.Linear(
            in_features=sizes[len(self.layer_sizes)],
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    std=1.0 / math.sqrt(sizes[len(self.layer_sizes)]))))
        self.add_sublayer("last_layer", self.last_layer)
        for i in range(len(layer_sizes)):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            act = paddle.nn.ReLU()
            self.add_sublayer('act_%d' % i, act)
            self._mlp_layers.append(act)
            if self.use_hidden_gate:
                self._hidden_gate_weight.append(
                    paddle.create_parameter(
                        shape=(sizes[i + 1], sizes[i + 1]),
                        dtype="float32",
                        name="hidden_gate_weight_%d" % i,
                        default_initializer=paddle.nn.initializer.Normal(
                            std=1.0 / math.sqrt(sizes[i + 1]))))
                self.add_parameter("hidden_gate_weight_%d" % i,
                                   self._hidden_gate_weight[i])

    def forward(self, sparse_inputs, dense_inputs):

        sparse_embs = []
        if self.use_embedding_gate:
            for i in range(len(self.embedding_gate_weight)):
                emb = self.embedding(sparse_inputs[i])
                emb = paddle.reshape(
                    emb, shape=[-1, self.sparse_feature_dim
                                ])  # emb shape [batchSize, sparse_feature_dim]
                gate = paddle.sum(paddle.multiply(
                    emb, self.embedding_gate_weight[i]),
                                  axis=-1,
                                  keepdim=True)  # gate shape [batchSize,1]
                activate_gate = paddle.nn.functional.sigmoid(
                    gate)  # activate_gate [batchSize,1]
                emb = paddle.multiply(
                    emb,
                    activate_gate)  # emb shape [batchSize, sparse_feature_dim]
                sparse_embs.append(emb)
        else:
            for s_input in sparse_inputs:
                emb = self.embedding(s_input)
                emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
                sparse_embs.append(emb)

        y_dnn = paddle.concat(x=sparse_embs + [dense_inputs], axis=1)

        for i in range(len(self._mlp_layers)):
            y_dnn = self._mlp_layers[i](y_dnn)
            if self.use_hidden_gate and i % 2 == 1 and i // 2 < len(
                    self._hidden_gate_weight):
                x_dnn = paddle.tanh(
                    paddle.matmul(y_dnn, self._hidden_gate_weight[i // 2]))
                y_dnn = paddle.multiply(y_dnn, x_dnn)
        y_dnn = self.last_layer(y_dnn)
        y_dnn = paddle.nn.functional.sigmoid(y_dnn)
        return y_dnn
