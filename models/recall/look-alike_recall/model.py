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
import paddle

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
        wl = paddle.create_parameter(
            shape=[self.user_emb_size, self.user_emb_size], dtype="float32")
        out = paddle.fluid.layers.matmul(user_seeds,
                                         wl)  # batch_size * max_len * emb_size
        out = paddle.fluid.layers.matmul(
            out, target_user, transpose_y=True)  # batch_size * max_len * 1
        out = paddle.tanh(out)
        out = paddle.nn.functional.softmax(x=out, axis=-2)
        out = user_seeds * out
        out = paddle.sum(x=out, axis=1)  # batch_size * emb_size
        return out

    def global_attention_unit(self, user_seeds):
        wg = paddle.create_parameter(
            shape=[self.user_emb_size, self.user_emb_size], dtype="float32")
        out = paddle.fluid.layers.matmul(user_seeds, wg)
        out = paddle.tanh(out)
        out = paddle.fluid.layers.matmul(out, user_seeds, transpose_y=True)
        out = paddle.nn.functional.softmax(x=out)
        out = paddle.fluid.layers.matmul(out, user_seeds)
        out = paddle.sum(x=out, axis=1)
        return out

    def net(self, inputs, is_infer=False):

        init_value_ = 0.1

        user_seeds = self._sparse_data_var[1]
        target_user = self._sparse_data_var[2]
        self.label = self._sparse_data_var[0]

        user_emb_attr = paddle.ParamAttr(name="user_emb")

        user_seeds_emb = paddle.static.nn.embedding(
            input=user_seeds,
            size=[self.user_count, self.user_emb_size],
            param_attr=user_emb_attr,
            is_sparse=True)

        target_user_emb = paddle.static.nn.embedding(
            input=target_user,
            size=[self.user_count, self.user_emb_size],
            param_attr=user_emb_attr,
            is_sparse=True)  # batch_size * 1 * emb_size
        user_seeds_emb = paddle.sum(x=user_seeds_emb,
                                    axis=1)  # batch_size(with lod) * emb_size

        pad_value = paddle.nn.functional.assign(input=np.array(
            [0.0], dtype=np.float32))
        user_seeds_emb, _ = paddle.fluid.layers.sequence_pad(
            user_seeds_emb, pad_value
        )  # batch_size(without lod) * max_sequence_length(in batch) * emb_size

        target_transform_matrix = paddle.create_parameter(
            shape=[self.user_emb_size, self.transformed_size], dtype="float32")
        seeds_transform_matrix = paddle.create_parameter(
            shape=[self.user_emb_size, self.transformed_size], dtype="float32")
        user_seeds_emb_transformed = paddle.fluid.layers.matmul(
            user_seeds_emb, seeds_transform_matrix)
        target_user_emb_transormed = paddle.fluid.layers.matmul(
            target_user_emb, target_transform_matrix)

        seeds_tower = self.local_attention_unit(
            user_seeds_emb_transformed,
            target_user_emb_transormed) + self.global_attention_unit(
                user_seeds_emb_transformed)

        target_tower = paddle.sum(x=target_user_emb_transormed, axis=1)

        score = paddle.fluid.layers.cos_sim(seeds_tower, target_tower)

        y_dnn = paddle.cast(self.label, dtype="float32")
        self.predict = paddle.nn.functional.sigmoid(score)
        cost = paddle.nn.functional.log_loss(
            input=score, label=paddle.cast(self.label, "float32"))
        avg_cost = paddle.sum(x=cost)

        self._cost = avg_cost

        predict_2d = paddle.concat(x=[1 - self.predict, self.predict], axis=1)
        label_int = paddle.cast(self.label, 'int64')
        auc_var, batch_auc_var, _ = paddle.fluid.layers.auc(input=predict_2d,
                                                            label=label_int,
                                                            slide_steps=0)
        self._metrics["AUC"] = auc_var
        self._metrics["BATCH_AUC"] = batch_auc_var
        if is_infer:
            self._infer_results["AUC"] = auc_var
