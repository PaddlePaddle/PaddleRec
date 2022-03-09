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


class FFMLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field):
        super(FFMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field

        self.ffm = FFM(sparse_feature_number, sparse_feature_dim,
                       dense_feature_dim, sparse_num_field)

        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, sparse_inputs, dense_inputs):

        y_first_order, y_second_order = self.ffm.forward(sparse_inputs,
                                                         dense_inputs)

        predict = F.sigmoid(y_first_order + y_second_order + self.bias)

        return predict


class FFM(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field):
        super(FFM, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.dense_emb_dim = self.sparse_feature_dim
        self.sparse_num_field = sparse_num_field
        self.init_value_ = 0.1

        use_sparse = True
        if paddle.is_compiled_with_npu():
            use_sparse = False

        # sparse part coding
        self.embedding_one = paddle.nn.Embedding(
            sparse_feature_number,
            1,
            sparse=use_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim * self.sparse_num_field,
            sparse=use_sparse,
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
            shape=[
                1, self.dense_feature_dim,
                self.dense_emb_dim * self.sparse_num_field
            ],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0))

    def forward(self, sparse_inputs, dense_inputs):
        # -------------------- first order term  --------------------
        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)
        sparse_emb_one = self.embedding_one(sparse_inputs_concat)

        dense_emb_one = paddle.multiply(dense_inputs, self.dense_w_one)
        dense_emb_one = paddle.unsqueeze(dense_emb_one, axis=2)

        y_first_order = paddle.sum(sparse_emb_one, 1) + paddle.sum(
            dense_emb_one, 1)

        # -------------------Field-aware second order term  --------------------
        sparse_embeddings = self.embedding(sparse_inputs_concat)
        dense_inputs_re = paddle.unsqueeze(dense_inputs, axis=2)
        dense_embeddings = paddle.multiply(dense_inputs_re, self.dense_w)
        feat_embeddings = paddle.concat([sparse_embeddings, dense_embeddings],
                                        1)

        field_aware_feat_embedding = paddle.reshape(
            feat_embeddings,
            shape=[
                -1, self.sparse_num_field, self.sparse_num_field,
                self.sparse_feature_dim
            ])
        field_aware_interaction_list = []
        for i in range(self.sparse_num_field):
            for j in range(i + 1, self.sparse_num_field):
                field_aware_interaction_list.append(
                    paddle.sum(field_aware_feat_embedding[:, i, j, :] *
                               field_aware_feat_embedding[:, j, i, :],
                               1,
                               keepdim=True))

        y_field_aware_second_order = paddle.add_n(field_aware_interaction_list)
        return y_first_order, y_field_aware_second_order
