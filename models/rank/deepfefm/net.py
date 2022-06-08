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
import itertools
from myutils import *


class DeepFEFMLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field, layer_sizes):
        super(DeepFEFMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.layer_sizes = layer_sizes

        self.fefm = FEFM(sparse_feature_number, sparse_feature_dim,
                         dense_feature_dim, sparse_num_field)
        self.dnn = DNN(sparse_feature_number, sparse_feature_dim,
                       dense_feature_dim, dense_feature_dim + sparse_num_field,
                       layer_sizes)
        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, sparse_inputs, dense_inputs):

        y_first_order, y_second_order, dnn_input = self.fefm(sparse_inputs,
                                                             dense_inputs)

        y_dnn = self.dnn(dnn_input)

        predict = F.sigmoid(y_first_order + y_second_order + y_dnn)

        return predict


class FEFM(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field):
        super(FEFM, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.dense_emb_dim = self.sparse_feature_dim
        self.sparse_num_field = sparse_num_field
        self.init_value_ = 0.1

        # sparse coding
        self.embedding_one = paddle.nn.Embedding(
            sparse_feature_number,
            1,
            padding_idx=0,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim))),
                regularizer=paddle.regularizer.L2Decay(1e-6)))

        # sparse embedding and dense embedding
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=False,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim))),
                regularizer=paddle.regularizer.L2Decay(1e-6)))

        # dense coding
        self.dense_w_one = paddle.create_parameter(
            shape=[self.dense_feature_dim],
            attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(1e-6)),
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0,
                std=self.init_value_ /
                math.sqrt(float(self.sparse_feature_dim))))

        # field embeddings
        self.field_embeddings = {}
        self.num_fields = self.sparse_num_field + self.dense_feature_dim
        for fi, fj in itertools.combinations(range(self.num_fields), 2):
            field_pair_id = str(fi) + "-" + str(fj)
            self.field_embeddings[field_pair_id] = paddle.create_parameter(
                shape=[self.sparse_feature_dim, self.sparse_feature_dim],
                attr=paddle.ParamAttr(
                    regularizer=paddle.regularizer.L2Decay(1e-7)),
                dtype='float32',
                default_initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim))))

    def forward(self, sparse_inputs, dense_inputs):
        # -------------------- first order term  --------------------
        sparse_inputs_concat = paddle.concat(
            sparse_inputs, axis=1)  # [batch_size, sparse_feature_number]
        sparse_emb_one = self.embedding_one(
            sparse_inputs_concat)  # [batch_size, sparse_feature_number, 1]

        _dense_emb_one = paddle.multiply(
            dense_inputs,
            self.dense_w_one)  # [batch_size, dense_feature_number]
        dense_emb_one = paddle.unsqueeze(
            _dense_emb_one, axis=2)  # [batch_size, dense_feature_number, 1]

        y_first_order = paddle.sum(sparse_emb_one, 1) + paddle.sum(
            dense_emb_one, 1)

        # -------------------- Field-embedded second order term  --------------------
        sparse_embeddings = self.embedding(
            sparse_inputs_concat
        )  # [batch_size, sparse_feature_number, sparse_feature_dim]
        dense_inputs_re = (dense_inputs * 1e5 + 1e6 + 2).astype(
            'int64')  # [batch_size, dense_feature_number]
        dense_embeddings = self.embedding(
            dense_inputs_re
        )  # [batch_size, dense_feature_number, dense_feature_dim]

        feat_embeddings = paddle.concat(
            [sparse_embeddings, dense_embeddings], 1
        )  # [batch_size, dense_feature_number + sparse_feature_number, dense_feature_dim]

        pairwise_inner_prods = []
        for fi, fj in itertools.combinations(
                range(self.num_fields), 2
        ):  # self.num_fields = 39, dense_feature_number + sparse_num_field
            field_pair_id = str(fi) + "-" + str(fj)
            feat_embed_i = paddle.squeeze(
                feat_embeddings[0:, fi:fi + 1, 0:], axis=1
            )  # feat_embeddings: [batch_size, num_fields, sparse_feature_dim]
            feat_embed_j = paddle.squeeze(
                feat_embeddings[0:, fj:fj + 1, 0:],
                axis=1)  # [batch_size * sparse_feature_dim]
            field_pair_embed_ij = self.field_embeddings[
                field_pair_id]  # self.field_embeddings [sparse_feature_dim, sparse_feature_dim]

            feat_embed_i_tr = paddle.matmul(
                feat_embed_i, field_pair_embed_ij + paddle.transpose(
                    field_pair_embed_ij,
                    [1, 0]))  # [batch_size * embedding_size]

            f = batch_dot(
                feat_embed_i_tr, feat_embed_j, axes=1)  # [batch_size * 1]
            pairwise_inner_prods.append(f)

        fefm_interaction_embedding = paddle.concat(
            pairwise_inner_prods,
            axis=1)  # [batch_size, num_fields*(num_fields-1)/2]

        y_field_emb_second_order = paddle.sum(fefm_interaction_embedding,
                                              axis=1,
                                              keepdim=True)

        dnn_input = paddle.reshape(sparse_embeddings, [0, -1])
        dnn_input = paddle.concat(
            [dnn_input, _dense_emb_one], 1
        )  # [batch_size, dense_feature_number + sparse_feature_number * sparse_feature_dim]
        dnn_input = paddle.concat(
            [dnn_input, fefm_interaction_embedding], 1
        )  # [batch_size, dense_feature_number + sparse_feature_number * sparse_feature_dim + num_fields*(num_fields-1)/2]

        return y_first_order, y_field_emb_second_order, dnn_input


class DNN(paddle.nn.Layer):
    def __init__(self,
                 sparse_feature_number,
                 sparse_feature_dim,
                 dense_feature_dim,
                 num_field,
                 layer_sizes,
                 dropout_rate=0.2):
        super(DNN, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes
        self.sparse_num_field = num_field - dense_feature_dim
        self.input_size = int(dense_feature_dim + self.sparse_num_field *
                              sparse_feature_dim + self.num_field * (
                                  self.num_field - 1) / 2)

        self.drop_out = paddle.nn.Dropout(p=dropout_rate)

        sizes = [self.input_size] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    regularizer=paddle.regularizer.L2Decay(1e-7),
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

    def forward(self, feat_embeddings):
        y_dnn = paddle.reshape(feat_embeddings, [-1, self.input_size])
        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
            y_dnn = self.drop_out(y_dnn)
        return y_dnn
