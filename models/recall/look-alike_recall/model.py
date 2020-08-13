#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.user_emb_size = envs.get_global_env(
            "hyper_parameters.user_emb_size", 64)
        self.user_count = envs.get_global_env("hyper_parameters.user_count",
                                              100000)
        self.transformed_size = envs.get_global_env(
            "hyper_parameter.transformed_size", 96)

    def local_attention_unit(self, user_seeds, target_user):
        wl = fluid.layers.create_parameter(
            shape=[self.user_emb_size, self.user_emb_size], dtype="float32")
        out = fluid.layers.matmul(user_seeds,
                                  wl)  # batch_size * max_len * emb_size
        out = fluid.layers.matmul(
            out, target_user, transpose_y=True)  # batch_size * max_len * 1
        out = fluid.layers.tanh(out)
        out = fluid.layers.softmax(out, axis=-2)
        out = user_seeds * out
        out = fluid.layers.reduce_sum(out, dim=1)  # batch_size * emb_size
        return out

    def global_attention_unit(self, user_seeds):
        wg = fluid.layers.create_parameter(
            shape=[self.user_emb_size, self.user_emb_size], dtype="float32")
        out = fluid.layers.matmul(user_seeds, wg)
        out = fluid.layers.tanh(out)
        out = fluid.layers.matmul(out, user_seeds, transpose_y=True)
        out = fluid.layers.softmax(out)
        out = fluid.layers.matmul(out, user_seeds)
        out = fluid.layers.reduce_sum(out, dim=1)
        return out

    def net(self, inputs, is_infer=False):

        init_value_ = 0.1

        user_seeds = self._sparse_data_var[1]
        target_user = self._sparse_data_var[2]
        self.label = self._sparse_data_var[0]

        user_emb_attr = fluid.ParamAttr(name="user_emb")

        user_seeds_emb = fluid.embedding(
            input=user_seeds,
            size=[self.user_count, self.user_emb_size],
            param_attr=user_emb_attr,
            is_sparse=True)

        target_user_emb = fluid.embedding(
            input=target_user,
            size=[self.user_count, self.user_emb_size],
            param_attr=user_emb_attr,
            is_sparse=True)  # batch_size * 1 * emb_size
        user_seeds_emb = fluid.layers.reduce_sum(
            user_seeds_emb, dim=1)  # batch_size(with lod) * emb_size

        pad_value = fluid.layers.assign(input=np.array(
            [0.0], dtype=np.float32))
        user_seeds_emb, _ = fluid.layers.sequence_pad(
            user_seeds_emb, pad_value
        )  # batch_size(without lod) * max_sequence_length(in batch) * emb_size

        target_transform_matrix = fluid.layers.create_parameter(
            shape=[self.user_emb_size, self.transformed_size], dtype="float32")
        seeds_transform_matrix = fluid.layers.create_parameter(
            shape=[self.user_emb_size, self.transformed_size], dtype="float32")
        user_seeds_emb_transformed = fluid.layers.matmul(
            user_seeds_emb, seeds_transform_matrix)
        target_user_emb_transormed = fluid.layers.matmul(
            target_user_emb, target_transform_matrix)

        seeds_tower = self.local_attention_unit(
            user_seeds_emb_transformed,
            target_user_emb_transormed) + self.global_attention_unit(
                user_seeds_emb_transformed)

        target_tower = fluid.layers.reduce_sum(
            target_user_emb_transormed, dim=1)

        score = fluid.layers.cos_sim(seeds_tower, target_tower)

        y_dnn = fluid.layers.cast(self.label, dtype="float32")
        self.predict = fluid.layers.sigmoid(score)
        cost = fluid.layers.log_loss(
            input=score, label=fluid.layers.cast(self.label, "float32"))
        avg_cost = fluid.layers.reduce_sum(cost)

        self._cost = avg_cost

        predict_2d = fluid.layers.concat([1 - self.predict, self.predict], 1)
        label_int = fluid.layers.cast(self.label, 'int64')
        auc_var, batch_auc_var, _ = fluid.layers.auc(input=predict_2d,
                                                     label=label_int,
                                                     slide_steps=0)
        self._metrics["AUC"] = auc_var
        self._metrics["BATCH_AUC"] = batch_auc_var
        if is_infer:
            self._infer_results["AUC"] = auc_var
