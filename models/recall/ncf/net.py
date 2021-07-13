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
import numpy as np
import math


class NCF_NeuMF_Layer(nn.Layer):
    def __init__(self, num_users, num_items, mf_dim, layers):
        super(NCF_NeuMF_Layer, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.layers = layers

        self.MF_Embedding_User = paddle.nn.Embedding(
            self.num_users,
            self.mf_dim,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.01),
                regularizer=paddle.regularizer.L2Decay(coeff=0)))
        self.MF_Embedding_Item = paddle.nn.Embedding(
            self.num_items,
            self.mf_dim,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.01),
                regularizer=paddle.regularizer.L2Decay(coeff=0)))
        self.MLP_Embedding_User = paddle.nn.Embedding(
            self.num_users,
            int(self.layers[0] / 2),
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.01),
                regularizer=paddle.regularizer.L2Decay(coeff=0)))
        self.MLP_Embedding_Item = paddle.nn.Embedding(
            self.num_items,
            int(self.layers[0] / 2),
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.01),
                regularizer=paddle.regularizer.L2Decay(coeff=0)))

        num_layer = len(self.layers)
        self.MLP_fc = []
        for i in range(1, num_layer):
            Linear = paddle.nn.Linear(
                in_features=self.layers[i - 1],
                out_features=self.layers[i],
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.TruncatedNormal(
                        mean=0.0, std=1.0 / math.sqrt(self.layers[i - 1])),
                    regularizer=paddle.regularizer.L2Decay(coeff=0)),
                name='layer_' + str(i))
            self.add_sublayer('layer_%d' % i, Linear)
            self.MLP_fc.append(Linear)
            act = paddle.nn.ReLU()
            self.add_sublayer('act_%d' % i, act)
            self.MLP_fc.append(act)

        self.prediction = paddle.nn.Linear(
            in_features=self.layers[2],
            out_features=1,
            weight_attr=nn.initializer.KaimingUniform(fan_in=self.layers[2] *
                                                      2),
            name='prediction')
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, input_data):
        user_input = input_data[0]
        item_input = input_data[1]
        label = input_data[2]

        # MF part
        user_embedding_mf = self.MF_Embedding_User(user_input)
        mf_user_latent = paddle.flatten(
            x=user_embedding_mf, start_axis=1, stop_axis=2)
        item_embedding_mf = self.MF_Embedding_Item(item_input)
        mf_item_latent = paddle.flatten(
            x=item_embedding_mf, start_axis=1, stop_axis=2)
        mf_vector = paddle.multiply(mf_user_latent, mf_item_latent)

        # MLP part
        # The 0-th layer is the concatenation of embedding layers
        user_embedding_mlp = self.MLP_Embedding_User(user_input)
        mlp_user_latent = paddle.flatten(
            x=user_embedding_mlp, start_axis=1, stop_axis=2)
        item_embedding_mlp = self.MLP_Embedding_Item(item_input)
        mlp_item_latent = paddle.flatten(
            x=item_embedding_mlp, start_axis=1, stop_axis=2)
        mlp_vector = paddle.concat(
            x=[mlp_user_latent, mlp_item_latent], axis=-1)

        for n_layer in self.MLP_fc:
            mlp_vector = n_layer(mlp_vector)

        # Concatenate MF and MLP parts
        predict_vector = paddle.concat(x=[mf_vector, mlp_vector], axis=-1)

        # Final prediction layer
        prediction = self.prediction(predict_vector)
        prediction = self.sigmoid(prediction)
        return prediction


class NCF_GMF_Layer(nn.Layer):
    def __init__(self, num_users, num_items, mf_dim, layers):
        super(NCF_GMF_Layer, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.layers = layers

        self.MF_Embedding_User = paddle.nn.Embedding(
            self.num_users,
            self.mf_dim,
            sparse=True,
            weight_attr=nn.initializer.Normal(
                mean=0.0, std=0.01))

        self.MF_Embedding_Item = paddle.nn.Embedding(
            self.num_items,
            self.mf_dim,
            sparse=True,
            weight_attr=nn.initializer.Normal(
                mean=0.0, std=0.01))

        self.prediction = paddle.nn.Linear(
            in_features=self.layers[3],
            out_features=1,
            weight_attr=nn.initializer.KaimingUniform(fan_in=None),
            name='prediction')

        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, input_data):

        user_input = input_data[0]
        item_input = input_data[1]
        label = input_data[2]

        user_embedding_mf = self.MF_Embedding_User(user_input)
        mf_user_latent = paddle.flatten(
            x=user_embedding_mf, start_axis=1, stop_axis=2)
        item_embedding_mf = self.MF_Embedding_Item(item_input)
        mf_item_latent = paddle.flatten(
            x=item_embedding_mf, start_axis=1, stop_axis=2)
        mf_vector = paddle.multiply(mf_user_latent, mf_item_latent)
        prediction = self.prediction(mf_vector)
        prediction = self.sigmoid(prediction)
        return prediction


class NCF_MLP_Layer(nn.Layer):
    def __init__(self, num_users, num_items, mf_dim, layers):
        super(NCF_MLP_Layer, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.layers = layers

        self.MLP_Embedding_User = paddle.nn.Embedding(
            self.num_users,
            int(self.layers[0] / 2),
            sparse=True,
            weight_attr=nn.initializer.Normal(
                mean=0.0, std=0.01))
        self.MLP_Embedding_Item = paddle.nn.Embedding(
            self.num_items,
            int(self.layers[0] / 2),
            sparse=True,
            weight_attr=nn.initializer.Normal(
                mean=0.0, std=0.01))

        num_layer = len(self.layers)
        self.MLP_fc = []
        for i in range(1, num_layer):
            Linear = paddle.nn.Linear(
                in_features=self.layers[i - 1],
                out_features=self.layers[i],
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.TruncatedNormal(
                        mean=0.0, std=1.0 / math.sqrt(self.layers[i - 1]))),
                name='layer_' + str(i))
            self.add_sublayer('layer_%d' % i, Linear)
            self.MLP_fc.append(Linear)
            act = paddle.nn.ReLU()
            self.add_sublayer('act_%d' % i, act)
            self.MLP_fc.append(act)

        self.prediction = paddle.nn.Linear(
            in_features=self.layers[3],
            out_features=1,
            weight_attr=nn.initializer.KaimingUniform(fan_in=self.layers[3] *
                                                      2),
            name='prediction')

        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, input_data):
        user_input = input_data[0]
        item_input = input_data[1]
        label = input_data[2]

        user_embedding_mlp = self.MLP_Embedding_User(user_input)
        mlp_user_latent = paddle.flatten(
            x=user_embedding_mlp, start_axis=1, stop_axis=2)
        item_embedding_mlp = self.MLP_Embedding_Item(item_input)
        mlp_item_latent = paddle.flatten(
            x=item_embedding_mlp, start_axis=1, stop_axis=2)
        mlp_vector = paddle.concat(
            x=[mlp_user_latent, mlp_item_latent], axis=-1)

        for n_layer in self.MLP_fc:
            mlp_vector = n_layer(mlp_vector)

        prediction = self.prediction(mlp_vector)
        prediction = self.sigmoid(prediction)
        return prediction
