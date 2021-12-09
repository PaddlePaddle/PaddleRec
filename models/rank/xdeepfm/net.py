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

from paddle.regularizer import L2Decay


class xDeepFMLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field, layer_sizes_cin,
                 layer_sizes_dnn):
        super(xDeepFMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.layer_sizes_cin = layer_sizes_cin
        self.layer_sizes_dnn = layer_sizes_dnn

        self.fm = Linear(sparse_feature_number, sparse_feature_dim,
                         dense_feature_dim, sparse_num_field)
        self.cin = CIN(sparse_feature_dim,
                       dense_feature_dim + sparse_num_field, layer_sizes_cin)

        self.dnn = DNN(sparse_feature_dim,
                       dense_feature_dim + sparse_num_field, layer_sizes_dnn)

        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, sparse_inputs, dense_inputs):

        y_linear, feat_embeddings = self.fm.forward(sparse_inputs,
                                                    dense_inputs)
        y_cin = self.cin.forward(feat_embeddings)
        y_dnn = self.dnn.forward(feat_embeddings)
        predict = F.sigmoid(y_linear + self.bias + y_cin + y_dnn)
        return predict


class Linear(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field):
        super(Linear, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.dense_emb_dim = self.sparse_feature_dim
        self.sparse_num_field = sparse_num_field
        self.init_value_ = 0.1

        # sparse part coding
        self.embedding_one = paddle.nn.Embedding(
            sparse_feature_number,
            1,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        # dense part coding
        self.dense_w_one = paddle.create_parameter(
            shape=[self.dense_feature_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0))

        self.dense_w = paddle.create_parameter(
            shape=[1, self.dense_feature_dim, self.dense_emb_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0))

    def forward(self, sparse_inputs, dense_inputs):
        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)
        sparse_emb_one = self.embedding_one(sparse_inputs_concat)

        dense_emb_one = paddle.multiply(dense_inputs, self.dense_w_one)
        dense_emb_one = paddle.unsqueeze(dense_emb_one, axis=2)

        y_linear = paddle.sum(sparse_emb_one, 1) + paddle.sum(dense_emb_one, 1)

        sparse_embeddings = self.embedding(sparse_inputs_concat)
        dense_inputs_re = paddle.unsqueeze(dense_inputs, axis=2)
        dense_embeddings = paddle.multiply(dense_inputs_re, self.dense_w)
        feat_embeddings = paddle.concat([sparse_embeddings, dense_embeddings],
                                        1)

        return y_linear, feat_embeddings


class CIN(nn.Layer):
    def __init__(self, sparse_feature_dim, num_field, layer_sizes_cin):
        super(CIN, self).__init__()
        self.sparse_feature_dim = sparse_feature_dim
        self.num_field = num_field
        self.layer_sizes_cin = layer_sizes_cin

        self.cnn_layers = []
        last_s = self.num_field
        for i in range(len(layer_sizes_cin)):
            _conv = nn.Conv2D(
                in_channels=last_s * self.num_field,
                out_channels=layer_sizes_cin[i],
                kernel_size=(1, 1),
                weight_attr=paddle.ParamAttr(
                    regularizer=L2Decay(coeff=0.0001),
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(last_s * self.num_field))),
                bias_attr=False)
            last_s = layer_sizes_cin[i]
            self.add_sublayer('cnn_%d' % i, _conv)
            self.cnn_layers.append(_conv)
        tmp_sum = sum(self.layer_sizes_cin)
        self.cin_linear = paddle.nn.Linear(
            in_features=tmp_sum,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                regularizer=L2Decay(coeff=0.0001),
                initializer=paddle.nn.initializer.Normal(std=0.1 /
                                                         math.sqrt(tmp_sum))))
        self.add_sublayer('cnn_fc', self.cin_linear)

    def forward(self, feat_embeddings):
        Xs = [feat_embeddings]
        last_s = self.num_field
        #m = paddle.nn.Dropout(p=0.5)

        for s, _conv in zip(self.layer_sizes_cin, self.cnn_layers):
            # calculate Z^(k+1) with X^k and X^0
            X_0 = paddle.reshape(
                x=paddle.transpose(Xs[0], [0, 2, 1]),
                shape=[-1, self.sparse_feature_dim, self.num_field,
                       1])  # None, embedding_size, num_field, 1
            X_k = paddle.reshape(
                x=paddle.transpose(Xs[-1], [0, 2, 1]),
                shape=[-1, self.sparse_feature_dim, 1,
                       last_s])  # None, embedding_size, 1, last_s
            Z_k_1 = paddle.matmul(
                x=X_0, y=X_k)  # None, embedding_size, num_field, last_s

            # compresses Z^(k+1) to X^(k+1)
            Z_k_1 = paddle.reshape(
                x=Z_k_1,
                shape=[-1, self.sparse_feature_dim, last_s * self.num_field
                       ])  # None, embedding_size, last_s*num_field
            Z_k_1 = paddle.transpose(
                Z_k_1, [0, 2, 1])  # None, s*num_field, embedding_size
            Z_k_1 = paddle.reshape(
                x=Z_k_1,
                shape=[
                    -1, last_s * self.num_field, 1, self.sparse_feature_dim
                ]
            )  # None, last_s*num_field, 1, embedding_size  (None, channal_in, h, w)

            X_k_1 = _conv(Z_k_1)

            X_k_1 = paddle.reshape(
                x=X_k_1,
                shape=[-1, s,
                       self.sparse_feature_dim])  # None, s, embedding_size
            #X_k_1 = m(X_k_1)
            Xs.append(X_k_1)
            last_s = s
        # sum pooling
        y_cin = paddle.concat(
            x=Xs[1:], axis=1)  # None, (num_field++), embedding_size
        y_cin = paddle.sum(x=y_cin, axis=-1)  # None, (num_field++)i
        tmp_sum = sum(self.layer_sizes_cin)
        y_cin = self.cin_linear(y_cin)
        y_cin = paddle.sum(x=y_cin, axis=-1, keepdim=True)

        return y_cin


class DNN(nn.Layer):
    def __init__(self, sparse_feature_dim, num_field, layer_sizes_dnn):
        super(DNN, self).__init__()
        self.sparse_feature_dim = sparse_feature_dim
        self.num_field = num_field
        self.layer_sizes_dnn = layer_sizes_dnn

        sizes = [sparse_feature_dim * num_field] + self.layer_sizes_dnn + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes_dnn))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes_dnn) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    regularizer=L2Decay(coeff=0.0001),
                    initializer=paddle.nn.initializer.Normal(
                        std=0.1 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)

    def forward(self, feat_embeddings):
        y_dnn = paddle.reshape(feat_embeddings,
                               [-1, self.num_field * self.sparse_feature_dim])
        #m = paddle.nn.Dropout(p=0.5)
        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
            #y_dnn = m(y_dnn)
        return y_dnn
