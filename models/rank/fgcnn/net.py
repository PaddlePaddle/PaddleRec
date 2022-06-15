# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.nn import functional as F
from itertools import combinations
import numpy as np
import pdb


class FGCNN(nn.Layer):
    def __init__(self, sparse_num_field, sparse_feature_size, feature_name,
                 feature_dim, dense_num_field, conv_kernel_width, conv_filters,
                 new_maps, pooling_width, stride, dnn_hidden_units,
                 dnn_dropout):
        '''
        Parameters
            vocab_size - 
        '''
        super(FGCNN, self).__init__()
        self.sparse_num_field = sparse_num_field
        self.dense_num_field = dense_num_field
        self.sparse_feature_size = sparse_feature_size
        self.feature_name = feature_name
        self.feature_dim = feature_dim
        self.feature_num_filed = self.sparse_num_field + self.dense_num_field
        self.conv_filters = conv_filters
        self.conv_kernel_width = conv_kernel_width
        self.new_maps = new_maps
        self.pooling_width = pooling_width
        self.stride = stride
        self.fg_embedding = nn.LayerList([
            EmbeddingLayer(
                num_embeddings=self.sparse_feature_size,
                embedding_dim=self.feature_dim,
                feature_name=self.feature_name[i] + '_fg_emd')
            for i in range(self.feature_num_filed)
        ])
        self.embedding = nn.LayerList([
            EmbeddingLayer(
                num_embeddings=self.sparse_feature_size,
                embedding_dim=self.feature_dim,
                feature_name=self.feature_name[i] + '_emd')
            for i in range(self.feature_num_filed)
        ])

        self.fgcnn = FGCNNLayer(self.feature_num_filed, self.feature_dim,
                                self.conv_filters, self.conv_kernel_width,
                                self.new_maps, self.pooling_width, self.stride)

        self.combined_feture_num = self.fgcnn.new_feture_num + self.feature_num_filed
        self.inner_product_layer = InnerProductLayer(self.combined_feture_num)
        self.dnn_input_dim = self.combined_feture_num * (self.combined_feture_num - 1) // 2\
                                + self.combined_feture_num * self.feature_dim

        self.dnn = DNNLayer(self.dnn_input_dim, dnn_hidden_units, dnn_dropout)

        self.fc_linear = self.add_sublayer(
            name='fc_linear',
            sublayer=nn.Linear(
                in_features=dnn_hidden_units[-1], out_features=1))

    def forward(self, inputs):
        inputs = paddle.to_tensor(inputs)
        fg_input_list = []
        origin_input_list = []
        for i in range(self.feature_num_filed):
            fg_input_list.append(self.fg_embedding[i](inputs[:, i].astype(
                'int64')).reshape((-1, 1, self.feature_dim)))
            origin_input_list.append(self.embedding[i](inputs[:, i].astype(
                'int64')).reshape((-1, 1, self.feature_dim)))
        fg_input = paddle.concat(fg_input_list, axis=1)
        origin_input = paddle.concat(origin_input_list, axis=1)
        new_features = self.fgcnn(fg_input)
        combined_input = paddle.concat([origin_input, new_features], axis=1)
        inner_product = self.inner_product_layer(combined_input)
        linear_signal = paddle.flatten(combined_input, start_axis=1)
        dnn_input = paddle.concat([linear_signal, inner_product], axis=1)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.fc_linear(dnn_output)
        y_pred = F.sigmoid(dnn_logit)
        return y_pred


class EmbeddingLayer(nn.Layer):
    def __init__(self, num_embeddings, embedding_dim, feature_name):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            name=feature_name,
            sparse=True)

    def forward(self, inputs):
        return self.embedding(inputs)


class FGCNNLayer(nn.Layer):
    def __init__(self, feature_num_field, embedding_size, filters,
                 kernel_width, new_maps, pooling_width, stride):
        super(FGCNNLayer, self).__init__()
        self.feature_num_field = feature_num_field
        self.embedding_size = embedding_size
        self.filters = filters
        self.kernel_width = kernel_width
        self.new_maps = new_maps
        self.pooling_width = pooling_width
        self.stride = stride
        self.init()
        # CNN network using tanh activation function and pooling layer
        self.conv_pooling = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(
                    in_channels=self.in_channels_size[i],
                    out_channels=self.filters[i],
                    kernel_size=(self.kernel_width[i], 1),
                    padding=(self.padding_size[i], 0),
                    stride=self.stride),
                nn.BatchNorm2D(self.filters[i]),
                nn.Tanh(),
                nn.MaxPool2D(
                    kernel_size=(self.pooling_width[i], 1),
                    stride=(self.pooling_width[i], 1)), )
            for i in range(len(self.filters))
        ])
        # fully connected layer to combine all the local features
        self.recombination = nn.LayerList([
            nn.Sequential(
                nn.Linear(
                    in_features=self.filters[i] * self.pooling_shape[i] *
                    self.embedding_size,
                    out_features=self.pooling_shape[i] * self.embedding_size *
                    self.new_maps[i],
                    name='fgcnn_linear_%d' % i),
                nn.Tanh()) for i in range(len(self.filters))
        ])

    def forward(self, inputs):
        # inputs shape: [batch_size, feature_num_field, embedding_size]
        feature = inputs.unsqueeze(1)
        # feature shape: [batch_size, 1, feature_num_field, embedding_size]
        new_feature_list = []
        for i in range(0, len(self.filters)):
            # use convolution layer to get new local feature
            feature = self.conv_pooling[i](feature)
            # use recombination layer to get new important features
            result = self.recombination[i](paddle.flatten(
                feature, start_axis=1))
            new_feature_list.append(
                paddle.reshape(
                    x=result,
                    shape=(-1, self.pooling_shape[i] * self.new_maps[i],
                           self.embedding_size)))
        new_features = paddle.concat(new_feature_list, axis=1)
        # new_features shape: [batch_size, new_feature_num, embedding_size]
        return new_features

    def init(self):
        # compute pooling shape
        self.pooling_shape = []
        self.pooling_shape.append(self.feature_num_field //
                                  self.pooling_width[0])
        for i in range(1, len(self.filters)):
            self.pooling_shape.append(self.pooling_shape[i - 1] //
                                      self.pooling_width[i])
        # compute padding size
        self.padding_size = []
        self.padding_size.append(
            ((self.feature_num_field - 1) * self.stride[0] +
             self.kernel_width[0] - self.feature_num_field) // 2)
        for i in range(1, len(self.filters)):
            self.padding_size.append(
                ((self.pooling_shape[i - 1] - 1) * self.stride[0] +
                 self.kernel_width[i] - self.pooling_shape[i - 1]) // 2)
        self.in_channels_size = [1, ] + list(self.filters)
        self.new_feture_num = sum([
            self.pooling_shape[i] * self.new_maps[i]
            for i in range(len(self.filters))
        ])


class DNNLayer(nn.Layer):
    def __init__(self, inputs_dim, hidden_units, dropout_rate):
        super(DNNLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

        hidden_units = [inputs_dim] + list(hidden_units)
        self.linears = nn.LayerList([
            nn.Sequential(
                nn.Linear(
                    in_features=hidden_units[i],
                    out_features=hidden_units[i + 1],
                    weight_attr=nn.initializer.Normal(
                        mean=0, std=1e-4),
                    name='dnn_%d' % i),
                nn.BatchNorm(hidden_units[i + 1])
                # nn.ReLU(hidden_units[i + 1],name='relu_%d' % i)
            ) for i in range(len(hidden_units) - 1)
        ])

        self.activation_layers = nn.LayerList([
            nn.ReLU(name='relu_%d' % i) for i in range(len(hidden_units) - 1)
        ])

    # @paddle.jit.to_static
    def forward(self, inputs):
        for i in range(len(self.linears)):
            inputs = self.linears[i](inputs)
            inputs = self.activation_layers[i](inputs)
            inputs = self.dropout(inputs)
        return inputs


class InnerProductLayer(nn.Layer):
    """ output: product_sum_pooling (bs x 1), 
                Bi_interaction_pooling (bs * dim), 
                inner_product (bs x f2/2), 
                elementwise_product (bs x f2/2 x emb_dim)
    """

    def __init__(self, num_fields=None):
        super(InnerProductLayer, self).__init__()
        if num_fields is None:
            raise ValueError("num_fields is required")
        else:
            self.num_fields = num_fields
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)

    def forward(self, feature_emb):
        onemask = paddle.ones(
            shape=[feature_emb.shape[0], self.num_fields, self.num_fields],
            dtype='int32')
        tri = paddle.triu(onemask, 1)
        upper_triange_mask = paddle.cast(tri, 'bool')
        inner_product_matrix = paddle.bmm(feature_emb,
                                          paddle.transpose(
                                              feature_emb, perm=[0, 2, 1]))
        flat_upper_triange = paddle.masked_select(inner_product_matrix,
                                                  upper_triange_mask)
        return flat_upper_triange.reshape([-1, self.interaction_units])
