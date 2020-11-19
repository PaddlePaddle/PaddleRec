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

import math
import paddle

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase
import numpy as np


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.num_users = envs.get_global_env("hyper_parameters.num_users")
        self.num_items = envs.get_global_env("hyper_parameters.num_items")
        self.latent_dim = envs.get_global_env("hyper_parameters.latent_dim")
        self.layers = envs.get_global_env("hyper_parameters.fc_layers")

    def input_data(self, is_infer=False, **kwargs):
        user_input = paddle.static.data(
            name="user_input", shape=[-1, 1], dtype="int64", lod_level=0)
        item_input = paddle.static.data(
            name="item_input", shape=[-1, 1], dtype="int64", lod_level=0)
        label = paddle.static.data(
            name="label", shape=[-1, 1], dtype="int64", lod_level=0)
        if is_infer:
            inputs = [user_input] + [item_input]
        else:
            inputs = [user_input] + [item_input] + [label]

        return inputs

    def net(self, inputs, is_infer=False):

        num_layer = len(self.layers)  # Number of layers in the MLP

        MF_Embedding_User = paddle.static.nn.embedding(
            input=inputs[0],
            size=[self.num_users, self.latent_dim],
            param_attr=paddle.fluid.initializer.Normal(
                loc=0.0, scale=0.01),
            is_sparse=True)
        MF_Embedding_Item = paddle.static.nn.embedding(
            input=inputs[1],
            size=[self.num_items, self.latent_dim],
            param_attr=paddle.fluid.initializer.Normal(
                loc=0.0, scale=0.01),
            is_sparse=True)

        MLP_Embedding_User = paddle.static.nn.embedding(
            input=inputs[0],
            size=[self.num_users, int(self.layers[0] / 2)],
            param_attr=paddle.fluid.initializer.Normal(
                loc=0.0, scale=0.01),
            is_sparse=True)
        MLP_Embedding_Item = paddle.static.nn.embedding(
            input=inputs[1],
            size=[self.num_items, int(self.layers[0] / 2)],
            param_attr=paddle.fluid.initializer.Normal(
                loc=0.0, scale=0.01),
            is_sparse=True)

        # MF part
        mf_user_latent = paddle.fluid.layers.flatten(
            x=MF_Embedding_User, axis=1)
        mf_item_latent = paddle.fluid.layers.flatten(
            x=MF_Embedding_Item, axis=1)
        mf_vector = paddle.multiply(x=mf_user_latent, y=mf_item_latent)

        # MLP part
        # The 0-th layer is the concatenation of embedding layers
        mlp_user_latent = paddle.fluid.layers.flatten(
            x=MLP_Embedding_User, axis=1)
        mlp_item_latent = paddle.fluid.layers.flatten(
            x=MLP_Embedding_Item, axis=1)
        mlp_vector = paddle.concat(
            x=[mlp_user_latent, mlp_item_latent], axis=-1)

        for i in range(1, num_layer):
            mlp_vector = paddle.static.nn.fc(
                x=mlp_vector,
                size=self.layers[i],
                activation='relu',
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.fluid.initializer.
                    TruncatedNormalInitializer(
                        loc=0.0, scale=1.0 / math.sqrt(mlp_vector.shape[1])),
                    regularizer=paddle.regularizer.L2Decay(coeff=1e-4)),
                name='layer_' + str(i))

        # Concatenate MF and MLP parts
        predict_vector = paddle.concat(x=[mf_vector, mlp_vector], axis=-1)

        # Final prediction layer
        prediction = paddle.static.nn.fc(
            x=predict_vector,
            size=1,
            activation='sigmoid',
            weight_attr=paddle.fluid.initializer.MSRAInitializer(uniform=True),
            name='prediction')
        if is_infer:
            self._infer_results["prediction"] = prediction
            return

        cost = paddle.nn.functional.log_loss(
            input=prediction, label=paddle.cast(
                x=inputs[2], dtype='float32'))
        avg_cost = paddle.mean(x=cost)

        self._cost = avg_cost
        self._metrics["cost"] = avg_cost
