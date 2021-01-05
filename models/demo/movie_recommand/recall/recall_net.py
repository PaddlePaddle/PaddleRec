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


class DNNLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim, fc_sizes):
        super(DNNLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.fc_sizes = fc_sizes

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            padding_idx=0,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        user_sizes = [36] + self.fc_sizes
        acts = ["relu" for _ in range(len(self.fc_sizes))]
        self._user_layers = []
        for i in range(len(self.fc_sizes)):
            linear = paddle.nn.Linear(
                in_features=user_sizes[i],
                out_features=user_sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(user_sizes[i]))))
            self.add_sublayer('linear_user_%d' % i, linear)
            self._user_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('user_act_%d' % i, act)
                self._user_layers.append(act)

        movie_sizes = [27] + self.fc_sizes
        acts = ["relu" for _ in range(len(self.fc_sizes))]
        self._movie_layers = []
        for i in range(len(self.fc_sizes)):
            linear = paddle.nn.Linear(
                in_features=movie_sizes[i],
                out_features=movie_sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(movie_sizes[i]))))
            self.add_sublayer('linear_movie_%d' % i, linear)
            self._movie_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('movie_act_%d' % i, act)
                self._movie_layers.append(act)

    def forward(self, batch_size, user_sparse_inputs, mov_sparse_inputs,
                label_input):

        user_sparse_embed_seq = []
        for s_input in user_sparse_inputs:
            emb = self.embedding(s_input)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            user_sparse_embed_seq.append(emb)

        mov_sparse_embed_seq = []
        for s_input in mov_sparse_inputs:
            s_input = paddle.reshape(s_input, shape=[batch_size, -1])
            emb = self.embedding(s_input)
            emb = paddle.sum(emb, axis=1)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            mov_sparse_embed_seq.append(emb)

        user_features = paddle.concat(user_sparse_embed_seq, axis=1)
        mov_features = paddle.concat(mov_sparse_embed_seq, axis=1)

        for n_layer in self._user_layers:
            user_features = n_layer(user_features)

        for n_layer in self._movie_layers:
            mov_features = n_layer(mov_features)

        sim = F.cosine_similarity(
            user_features, mov_features, axis=1).reshape([-1, 1])
        predict = paddle.scale(sim, scale=5)

        return predict
