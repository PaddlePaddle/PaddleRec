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
import paddle.nn.functional as fun
import math


class MLPLayer(nn.Layer):
    def __init__(self, input_shape, units_list=None, activation=None,
                 **kwargs):
        super(MLPLayer, self).__init__(**kwargs)

        if units_list is None:
            units_list = [128, 128, 64]
        units_list = [input_shape] + units_list

        self.units_list = units_list
        self.mlp = []
        self.activation = activation

        for i, unit in enumerate(units_list[:-1]):
            if i != len(units_list) - 1:
                dense = paddle.nn.Linear(
                    in_features=unit,
                    out_features=units_list[i + 1],
                    weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.TruncatedNormal(
                            std=1.0 / math.sqrt(unit))))
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                relu = paddle.nn.ReLU()
                self.mlp.append(relu)
                self.add_sublayer('relu_%d' % i, relu)

                norm = paddle.nn.BatchNorm1D(units_list[i + 1])
                self.mlp.append(norm)
                self.add_sublayer('norm_%d' % i, norm)
            else:
                dense = paddle.nn.Linear(
                    in_features=unit,
                    out_features=units_list[i + 1],
                    weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.TruncatedNormal(
                            std=1.0 / math.sqrt(unit))))
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                if self.activation is not None:
                    relu = paddle.nn.ReLU()
                    self.mlp.append(relu)
                    self.add_sublayer('relu_%d' % i, relu)

    def forward(self, inputs):
        outputs = inputs
        for n_layer in self.mlp:
            outputs = n_layer(outputs)
        return outputs


class FENLayer(nn.Layer):
    def __init__(self, sparse_field_num, sparse_feature_num,
                 sparse_feature_dim, dense_feature_dim, fen_layers_size,
                 dense_layers_size):
        super(FENLayer, self).__init__()
        self.sparse_field_num = sparse_field_num
        self.sparse_feature_num = sparse_feature_num
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.fen_layers_size = fen_layers_size
        self.dense_layers_size = dense_layers_size

        self.fen_mlp = MLPLayer(
            input_shape=(sparse_field_num + 1) * sparse_feature_dim,
            units_list=fen_layers_size,
            activation="relu")

        use_sparse = True
        if paddle.is_compiled_with_custom_device('npu'):
            use_sparse = False

        self.sparse_embedding = paddle.nn.Embedding(
            num_embeddings=self.sparse_feature_num,
            embedding_dim=self.sparse_feature_dim,
            sparse=use_sparse,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))
        self.sparse_weight = paddle.nn.Embedding(
            num_embeddings=self.sparse_feature_num,
            embedding_dim=1,
            sparse=use_sparse,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors2",
                initializer=paddle.nn.initializer.Uniform()))
        self.dense_linear = paddle.nn.Linear(
            in_features=self.dense_feature_dim, out_features=1)
        self.dense_mlp = MLPLayer(
            input_shape=self.dense_feature_dim,
            units_list=self.dense_layers_size,
            activation="relu")
        self.dnn_mlp = MLPLayer(
            input_shape=self.sparse_feature_dim,
            units_list=[1],
            activation="relu")

    def forward(self, sparse_inputs, dense_inputs):
        # ------------------ first order ------------------------------------
        # (batch_size, sparse_field_num)
        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)

        # (batch_size, sparse_field_num, 1)
        sparse_emb_one = self.sparse_weight(sparse_inputs_concat)

        # (batch_size, sparse_field_num)
        sparse_emb_one = paddle.squeeze(sparse_emb_one, axis=-1)

        # (batch_size, 1)
        dense_emb_one = self.dense_linear(dense_inputs)

        # (batch_size, (sparse_field_num + 1))
        feat_emb_one = paddle.concat([dense_emb_one, sparse_emb_one], axis=1)

        # -------------------- fen layer ------------------------------------
        # (batch_size, embedding_size)
        dense_embedding = self.dense_mlp(dense_inputs)
        dnn_logits = self.dnn_mlp(dense_embedding)
        dense_embedding = paddle.unsqueeze(dense_embedding, axis=1)

        # (batch_size, sparse_field_num, embedding_size)
        sparse_embedding = self.sparse_embedding(sparse_inputs_concat)

        # (batch_size, (sparse_field_num + 1), embedding_size)
        feat_embeddings = paddle.concat(
            [dense_embedding, sparse_embedding], axis=1)
        batch_size, sparse_field_num_1, embedding_size = feat_embeddings.shape

        # (batch_size, (sparse_field_num + 1))
        m_x = self.fen_mlp(
            paddle.reshape(
                feat_embeddings,
                shape=(batch_size, sparse_field_num_1 * embedding_size)))
        return dnn_logits, feat_emb_one, feat_embeddings, m_x


class FMLayer(nn.Layer):
    def __init__(self):
        super(FMLayer, self).__init__()
        self.bias = paddle.create_parameter(
            is_bias=True, shape=[1], dtype='float32')

    def forward(self, dnn_logits, first_order, combined_features):
        """
        first_order: FM first order (batch_size, 1)
        combined_features: FM sparse features (batch_size, sparse_field_num + 1, embedding_size)
        """
        # sum square part
        # (batch_size, embedding_size)
        summed_features_emb = paddle.sum(combined_features, axis=1)
        summed_features_emb_square = paddle.square(summed_features_emb)

        # square sum part
        squared_features_emb = paddle.square(combined_features)

        # (batch_size, embedding_size)
        squared_sum_features_emb = paddle.sum(squared_features_emb, axis=1)

        # (batch_size, 1)
        logits = first_order + 0.5 * paddle.sum(
            summed_features_emb_square - squared_sum_features_emb,
            axis=1,
            keepdim=True) + self.bias + dnn_logits
        return fun.sigmoid(logits)


class MultiHeadAttentionLayer(nn.Layer):
    def __init__(self, att_factor_dim, att_head_num, sparse_feature_dim,
                 sparse_field_num):
        super(MultiHeadAttentionLayer, self).__init__()
        self.att_factor_dim = att_factor_dim
        self.att_head_num = att_head_num
        self.sparse_feature_dim = sparse_feature_dim
        self.sparse_field_num = sparse_field_num

        self.W_Query = paddle.create_parameter(
            default_initializer=nn.initializer.TruncatedNormal(),
            shape=[
                self.sparse_feature_dim,
                self.att_factor_dim * self.att_head_num
            ],
            dtype='float32')
        self.W_Key = paddle.create_parameter(
            default_initializer=nn.initializer.TruncatedNormal(),
            shape=[
                self.sparse_feature_dim,
                self.att_factor_dim * self.att_head_num
            ],
            dtype='float32')
        self.W_Value = paddle.create_parameter(
            default_initializer=nn.initializer.TruncatedNormal(),
            shape=[
                self.sparse_feature_dim,
                self.att_factor_dim * self.att_head_num
            ],
            dtype='float32')
        self.W_Res = paddle.create_parameter(
            default_initializer=nn.initializer.TruncatedNormal(),
            shape=[
                self.sparse_feature_dim,
                self.att_factor_dim * self.att_head_num
            ],
            dtype='float32')
        self.dnn_layer = MLPLayer(
            input_shape=(self.sparse_field_num + 1) * self.att_factor_dim *
            self.att_head_num,
            units_list=[self.sparse_field_num + 1],
            activation="relu")

    def forward(self, combined_features):
        """
        combined_features: (batch_size, (sparse_field_num + 1), embedding_size)
        W_Query: (embedding_size, factor_dim * att_head_num)
        (b, f, e) * (e, d*h) -> (b, f, d*h)
        """
        # (b, f, d*h)
        querys = paddle.matmul(combined_features, self.W_Query)
        keys = paddle.matmul(combined_features, self.W_Key)
        values = paddle.matmul(combined_features, self.W_Value)
        b, f, d_h = querys.shape

        # (h, b, f, d) <- (b, f, d)
        querys = paddle.stack(paddle.split(querys, self.att_head_num, axis=2))
        keys = paddle.stack(paddle.split(keys, self.att_head_num, axis=2))
        values = paddle.stack(paddle.split(values, self.att_head_num, axis=2))

        # (h, b, f, f)
        inner_product = paddle.matmul(querys, keys, transpose_y=True)
        inner_product /= self.att_factor_dim**0.5
        normalized_att_scores = fun.softmax(inner_product)

        # (h, b, f, d)
        result = paddle.matmul(normalized_att_scores, values)
        result = paddle.concat(
            paddle.split(
                result, self.att_head_num, axis=0), axis=-1)

        # (b, f, h * d)
        result = paddle.squeeze(result, axis=0)
        result += paddle.matmul(combined_features, self.W_Res)

        # (b, f * h * d)
        result = paddle.reshape(result, shape=(b, f * d_h))
        m_vec = self.dnn_layer(result)
        return m_vec


class IFM(nn.Layer):
    def __init__(self, sparse_field_num, sparse_feature_num,
                 sparse_feature_dim, dense_feature_dim, fen_layers_size,
                 dense_layers_size):
        super(IFM, self).__init__()
        self.sparse_field_num = sparse_field_num
        self.sparse_feature_num = sparse_feature_num
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.fen_layers_size = fen_layers_size
        self.dense_layers_size = dense_layers_size
        self.fen_layer = FENLayer(
            sparse_field_num=self.sparse_field_num,
            sparse_feature_num=self.sparse_feature_num,
            sparse_feature_dim=self.sparse_feature_dim,
            dense_feature_dim=self.dense_feature_dim,
            fen_layers_size=self.fen_layers_size,
            dense_layers_size=self.dense_layers_size)
        self.fm_layer = FMLayer()

    def forward(self, sparse_inputs, dense_inputs):
        dnn_logits, feat_emb_one, feat_embeddings, m_x = self.fen_layer(
            sparse_inputs, dense_inputs)

        m_x = fun.softmax(m_x)

        # (batch_size, (sparse_field_num + 1))
        feat_emb_one = feat_emb_one * m_x

        # (batch_size, (sparse_field_num + 1), embedding_size)
        feat_embeddings = feat_embeddings * paddle.unsqueeze(m_x, axis=-1)

        # (batch_size, 1)
        first_order = paddle.sum(feat_emb_one, axis=1, keepdim=True)

        return self.fm_layer(dnn_logits, first_order, feat_embeddings)


class DIFM(nn.Layer):
    def __init__(self, sparse_field_num, sparse_feature_num,
                 sparse_feature_dim, dense_feature_dim, fen_layers_size,
                 dense_layers_size, att_factor_dim, att_head_num):
        super(DIFM, self).__init__()
        self.sparse_field_num = sparse_field_num
        self.sparse_feature_num = sparse_feature_num
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.fen_layers_size = fen_layers_size
        self.dense_layers_size = dense_layers_size
        self.att_factor_dim = att_factor_dim
        self.att_head_num = att_head_num

        self.fen_layer = FENLayer(
            sparse_field_num=self.sparse_field_num,
            sparse_feature_num=self.sparse_feature_num,
            sparse_feature_dim=self.sparse_feature_dim,
            dense_feature_dim=self.dense_feature_dim,
            fen_layers_size=self.fen_layers_size,
            dense_layers_size=self.dense_layers_size)
        self.fm_layer = FMLayer()
        self.mha_layer = MultiHeadAttentionLayer(
            att_factor_dim=self.att_factor_dim,
            att_head_num=self.att_head_num,
            sparse_feature_dim=self.sparse_feature_dim,
            sparse_field_num=self.sparse_field_num)

    def forward(self, sparse_inputs, dense_inputs):
        dnn_logits, feat_emb_one, feat_embeddings, m_bit = self.fen_layer(
            sparse_inputs, dense_inputs)
        m_vec = self.mha_layer(feat_embeddings)
        m = fun.softmax(m_vec + m_bit)

        feat_emb_one = feat_emb_one * m
        feat_embeddings = feat_embeddings * paddle.unsqueeze(m, axis=-1)

        first_order = paddle.sum(feat_emb_one, axis=1, keepdim=True)

        return self.fm_layer(dnn_logits, first_order, feat_embeddings)
