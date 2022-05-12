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
import numpy as np
import copy
import math


class PositionalEncoder(nn.Layer):
    def __init__(self, d_model, max_seq_len=50):
        #d_model为嵌入维度
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model

        position = np.array([[
            pos / np.power(10000, 2. * i / self.d_model)
            for i in range(self.d_model)
        ] for pos in range(max_seq_len)])
        # Second part, apply the cosine to even columns and sin to odds.
        position[:, 0::2] = np.sin(position[:, 0::2])  # dim 2i
        position[:, 1::2] = np.cos(position[:, 1::2])  # dim 2i+1
        self.position = self.create_parameter(
            shape=[max_seq_len, self.d_model],
            default_initializer=paddle.nn.initializer.Assign(value=position))

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.shape[1]
        x = x + self.position[:seq_len, :]
        return x


class AttentionSequencePoolingLayer(nn.Layer):
    def __init__(self,
                 dnn_units=[8, 64, 16],
                 dnn_activation='sigmoid',
                 weight_normalization=False,
                 name=None):
        super().__init__()
        self.dnn_units = dnn_units
        self.dnn_activation = 'sigmoid'
        self.weight_normalization = weight_normalization
        self.name = name
        layer_list = []
        #bn_list = []
        for i in range(len(dnn_units) - 1):
            dnn_layer = nn.Linear(
                in_features=self.dnn_units[i]
                if i != 0 else self.dnn_units[i] * 4,
                out_features=self.dnn_units[i + 1],
                weight_attr=self._weight_init())
            self.add_sublayer(self.name + f'linear_{i}', dnn_layer)
            layer_list.append(dnn_layer)
            #layer_list.append(copy.deepcopy(dnn_layer))
            #bn_layer = nn.BatchNorm(50)
            #self.add_sublayer(self.name + f'bn_{i}', bn_layer)
            #bn_list.append(bn_layer)
            #bn_list.append(copy.deepcopy(bn_layer))
        #self.bn_layer = nn.LayerList(bn_list)
        self.layers = nn.LayerList(layer_list)
        self.dnn = nn.Linear(
            self.dnn_units[-1], 1, weight_attr=self._weight_init())
        self.activation = nn.Sigmoid()
        self.soft = nn.Softmax()

    def _weight_init(self):
        return paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.XavierNormal())

    def forward(self, inputs):
        querys, keys, sess_length = inputs
        #assert(type(sess_length) == paddle.Tensor), f"At Attention SequencePoolingLayer expected inputs[2]'s type is paddle.Tensor, but got {type(sess_length)}"
        keys_length = keys.shape[1]
        key_masks = nn.functional.sequence_mask(sess_length, keys_length)
        querys = paddle.tile(querys.unsqueeze(1), [1, keys_length, 1])
        att_input = paddle.concat(
            [querys, keys, querys - keys, querys * keys], axis=-1)
        for i, layer in enumerate(self.layers):
            att_input = layer(att_input)
            #att_input = self.bn_layer[i](att_input)  # BatchNomalization
            att_input = self.activation(att_input)  # activation 
        att_score = self.dnn(att_input)  # (N, 50, 1)
        att_score = paddle.transpose(att_score, [0, 2, 1])  # (N, 1, 50)
        if self.weight_normalization:
            paddings = paddle.ones_like(att_score) * (-2**32 + 1)
        else:
            paddings = paddle.zeros_like(att_score)
        att_score = paddle.where(
            key_masks.unsqueeze(1) == 1, att_score, paddings
        )  # key_masks.unsqueeze in order to keep shape same as att_score
        att_score = self.soft(att_score)
        out = paddle.matmul(att_score, keys)
        return out


class MLP(nn.Layer):
    def __init__(self, mlp_hidden_units, use_bn=True):
        super().__init__()
        self.mlp_hidden_units = mlp_hidden_units
        self.acitivation = paddle.nn.Sigmoid()
        layer_list = []
        for i in range(len(mlp_hidden_units) - 1):
            dnn_layer = nn.Linear(
                in_features=self.mlp_hidden_units[i],
                out_features=self.mlp_hidden_units[i + 1],
                weight_attr=self._weight_init())
            self.add_sublayer(f'linear_{i}', dnn_layer)
            layer_list.append(dnn_layer)
        self.layers = nn.LayerList(layer_list)
        self.dense = nn.Linear(
            self.mlp_hidden_units[-1],
            1,
            bias_attr=True,
            weight_attr=self._weight_init())
        self.predict_layer = nn.Sigmoid()

    def _weight_init(self):
        return paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.XavierNormal())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.acitivation(x)
        x = self.dense(x)
        x = self.predict_layer(x)
        return x
