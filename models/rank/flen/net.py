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

from paddle.regularizer import L2Decay


class FLENLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 sparse_inputs_slots, sparse_num_field, layer_sizes_dnn):
        super(FLENLayer, self).__init__()

        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.sparse_num_field = sparse_num_field
        self.sparse_inputs_slots = sparse_inputs_slots
        self.layer_sizes_dnn = layer_sizes_dnn

        self._EmbeddingLayer = EmbeddingLayer(
            sparse_feature_number, sparse_feature_dim, sparse_num_field)
        self._DNNLayer = DNNLayer(sparse_feature_dim, sparse_inputs_slots,
                                  layer_sizes_dnn)
        self._FieldWiseBiInteraction = FieldWiseBiInteraction(
            sparse_feature_dim, sparse_num_field)

        self.fwbi_fc_32 = paddle.nn.Linear(
            in_features=self.sparse_feature_dim,
            out_features=self.sparse_feature_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))

        self.add_sublayer('fwbi_fc_32', self.fwbi_fc_32)

        self.fwbi_relu = paddle.nn.ReLU()
        self.add_sublayer('fwbi_relu', self.fwbi_relu)

        self.fwbi_bn = paddle.nn.BatchNorm1D(self.sparse_feature_dim)
        self.add_sublayer('fwbi_bn', self.fwbi_bn)

        self.fwbi_drop = paddle.nn.Dropout(p=0.2)

        self.linear = paddle.nn.Linear(
            in_features=2 * self.sparse_feature_dim,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))

        self.add_sublayer('linear_out', self.linear)

    def forward(self, sparse_inputs):
        user_inputs = sparse_inputs[1:14]
        item_inputs = sparse_inputs[14:17]
        contex_inputs = sparse_inputs[17:]

        # Embedding 
        field_wise_embed_list = []
        for inputs in [user_inputs, item_inputs, contex_inputs]:

            field_emb = self._EmbeddingLayer(inputs)
            field_wise_embed_list.append(field_emb)

        # mlp part
        dnn_input = paddle.concat(field_wise_embed_list, axis=1)
        dnn_output = self._DNNLayer(dnn_input)

        # field-weighted embedding
        fm_mf_out = self._FieldWiseBiInteraction(field_wise_embed_list)
        fwbi_fc_32 = self.fwbi_fc_32(fm_mf_out)
        fwbi_fc_32 = self.fwbi_relu(fwbi_fc_32)
        fwbi_bn = self.fwbi_bn(fwbi_fc_32)
        fwbi_drop = self.fwbi_drop(fwbi_bn)

        logits = paddle.concat(
            [fwbi_drop, dnn_output], axis=1)  # [bacth, 2*sparse_feature_dim]

        y = self.linear(logits)
        predict = F.sigmoid(y)

        return predict


class EmbeddingLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 sparse_num_field):
        super(EmbeddingLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.sparse_num_field = sparse_num_field

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))

    def forward(self, sparse_inputs):
        emb_list = []
        for data in sparse_inputs:
            feat_emb = self.embedding(data)
            emb_list.append(feat_emb)
        field_emb = paddle.concat(emb_list, axis=1)

        return field_emb


class DNNLayer(nn.Layer):
    def __init__(self,
                 sparse_feature_dim,
                 sparse_inputs_slots,
                 layer_sizes_dnn,
                 dropout_rate=0.2):
        super(DNNLayer, self).__init__()
        self.sparse_feature_dim = sparse_feature_dim
        self.num_field = sparse_inputs_slots
        self.layer_sizes_dnn = layer_sizes_dnn
        self.drop = paddle.nn.Dropout(p=dropout_rate)

        sizes = [sparse_feature_dim * self.num_field] + self.layer_sizes_dnn

        self._mlp_layers = []

        for i in range(len(layer_sizes_dnn)):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()))

            self._mlp_layers.append(linear)
            self.add_sublayer('linear_%d' % i, linear)

            relu = paddle.nn.ReLU()
            self._mlp_layers.append(relu)
            self.add_sublayer('relu_%d' % i, relu)

            norm = paddle.nn.BatchNorm1D(sizes[i + 1])
            self._mlp_layers.append(norm)
            self.add_sublayer('norm_%d' % i, norm)

    def forward(self, feat_embeddings):
        y_dnn = paddle.reshape(feat_embeddings,
                               [-1, self.num_field * self.sparse_feature_dim])

        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
            y_dnn = self.drop(y_dnn)
        return y_dnn


class FieldWiseBiInteraction(nn.Layer):
    def __init__(self,
                 sparse_feature_dim,
                 num_fields,
                 activation=None,
                 use_bias=False):
        super(FieldWiseBiInteraction, self).__init__()
        self.sparse_feature_dim = sparse_feature_dim
        self.num_fields = num_fields
        self.use_bias = use_bias
        self.activation = activation

        self.kernel_mf = paddle.create_parameter(
            shape=[int(self.num_fields * (self.num_fields - 1) / 2), 1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.XavierUniform())

        self.kernel_fm = paddle.create_parameter(
            shape=[self.num_fields, 1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.XavierUniform())

        if self.use_bias:
            self.bias_mf = paddle.create_parameter(
                shape=[1, ],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(value=0.0))

            self.bias_fm = paddle.create_parameter(
                shape=[1, ],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, inputs):

        fields_wise_embeds_list = inputs

        # MF module
        field_wise_vectors = paddle.concat(
            [
                paddle.sum(fields_i_vectors, axis=1, keepdim=True)
                for fields_i_vectors in fields_wise_embeds_list
            ],
            1)

        left = []
        right = []

        for i, j in itertools.combinations(list(range(self.num_fields)), 2):
            left.append(i)
            right.append(j)

        left = paddle.to_tensor(left)
        right = paddle.to_tensor(right)

        embeddings_left = paddle.gather(field_wise_vectors, index=left, axis=1)
        embeddings_right = paddle.gather(
            field_wise_vectors, index=right, axis=1)

        embeddings_prod = paddle.multiply(embeddings_left, embeddings_right)
        field_weighted_embedding = paddle.multiply(embeddings_prod,
                                                   self.kernel_mf)
        h_mf = paddle.sum(field_weighted_embedding, axis=1)

        if self.use_bias:
            h_mf = h_mf + self.bias_mf

        # FM module
        square_of_sum_list = [
            paddle.square(paddle.sum(field_i_vectors, axis=1, keepdim=True))
            for field_i_vectors in fields_wise_embeds_list
        ]

        sum_of_square_list = [
            paddle.sum(paddle.multiply(field_i_vectors, field_i_vectors),
                       axis=1,
                       keepdim=True)
            for field_i_vectors in fields_wise_embeds_list
        ]

        field_fm = paddle.concat([
            square_of_sum - sum_of_square
            for square_of_sum, sum_of_square in zip(square_of_sum_list,
                                                    sum_of_square_list)
        ], 1)
        h_fm = paddle.sum(paddle.multiply(field_fm, self.kernel_fm), axis=1)

        if self.use_bias:
            h_fm = h_fm + self.bias_fm

        return h_mf
