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
import paddle.fluid as fluid
import paddle.nn.functional as F
import math


class DNNLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes, use_embedding_gate, use_hidden_gate):
        super(DNNLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes
        self.use_embedding_gate = use_embedding_gate
        self.use_hidden_gate = use_hidden_gate
        if self.use_embedding_gate:
            self.embedding_gate_weight = [fluid.layers.create_parameter(shape=[1], dtype="float32", name='embedding_gate_weight_%d' % i) for i in range(num_field)]
            for i in range(num_field):
                self.add_parameter('embedding_gate_weight_%d' % i, self.embedding_gate_weight[i])
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        sizes = [sparse_feature_dim * num_field + dense_feature_dim
                 ] + self.layer_sizes
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        self._hidden_gate_layers = []
        self.last_layer = paddle.nn.Linear(
                in_features=sizes[len(self.layer_sizes)],
                out_features=1,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[len(self.layer_sizes)]))))
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
                hidden_linear = paddle.nn.Linear(
                    in_features=sizes[i + 1],
                    out_features=sizes[i + 1],
                    weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Normal(
                            std=1.0 / math.sqrt(sizes[i + 1]))))
                self.add_sublayer('hidden_linear_%d' % i, hidden_linear)
                self._hidden_gate_layers.append(hidden_linear)
            self.add_sublayer("last_layer",self.last_layer)


    def forward(self, sparse_inputs, dense_inputs):

        sparse_embs = []

        if self.use_embedding_gate:
            for i in range(len(self.embedding_gate_weight)):
                emb = self.embedding(sparse_inputs[i])
                emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
                # emb shape [batchSize, sparse_feature_dim]
                gate = fluid.layers.reduce_sum(fluid.layers.elementwise_mul(emb, self.embedding_gate_weight[i]), dim=-1)
                # gate shape [batchSize]
                activate_gate = fluid.layers.sigmoid(gate)
                # activate_gate [batchSize]
                emb = fluid.layers.elementwise_mul(emb, activate_gate, axis=0)
                # emb shape [batchSize, sparse_feature_dim]
                sparse_embs.append(emb)
        else:
            for s_input in sparse_inputs:
                emb = self.embedding(s_input)
                emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
                sparse_embs.append(emb)


        y_dnn = paddle.concat(x=sparse_embs + [dense_inputs], axis=1)

        for i in range(len(self._mlp_layers)):
            y_dnn = self._mlp_layers[i](y_dnn)
            if self.use_hidden_gate and i % 2 == 1 and i // 2 < len(self._hidden_gate_layers):
                x_dnn = fluid.layers.tanh(self._hidden_gate_layers[i // 2](y_dnn))
                y_dnn = fluid.layers.elementwise_mul(y_dnn, x_dnn)
        y_dnn = self.last_layer(y_dnn)
        y_dnn = fluid.layers.sigmoid(y_dnn)
        return y_dnn
