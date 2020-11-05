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
import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.dense_feature_dim = envs.get_global_env(
            "hyper_parameters.dense_feature_dim")
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim")
        self.learning_rate = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate")

    def input_data(self, is_infer=False, **kwargs):
        q = fluid.layers.data(
            name="query", shape=[1], dtype="int64", lod_level=1)
        pt = fluid.layers.data(
            name="pos_title", shape=[1], dtype="int64", lod_level=1)
        nt = fluid.layers.data(
            name="neg_title", shape=[1], dtype="int64", lod_level=1)

        inputs = [q, pt, nt]
        return inputs

    def net(self, input, is_infer=False):
        dict_dim = self.dict_dim
        emb_dim = self.emb_dim
        hid_dim = self.hid_dim
        base_lr = self.learning_rate
        emb_lr = self.learning_rate * 3

        q = input[0]
        pt = input[1]
        nt = input[2]

        q_emb = fluid.layers.embedding(
            input=q,
            size=[dict_dim, emb_dim],
            param_attr=fluid.ParamAttr(
                name="__emb__", learning_rate=emb_lr),
            is_sparse=is_sparse)
        # vsum
        q_sum = fluid.layers.sequence_pool(input=q_emb, pool_type='sum')
        q_ss = fluid.layers.softsign(q_sum)
        # fc layer after conv
        q_fc = fluid.layers.fc(input=q_ss,
                               size=hid_dim,
                               param_attr=fluid.ParamAttr(
                                   name="__q_fc__",
                                   learning_rate=base_lr,
                                   initializer=fluid.initializer.Xavier()))
        # embedding
        pt_emb = fluid.layers.embedding(
            input=pt,
            size=[dict_dim, emb_dim],
            param_attr=fluid.ParamAttr(
                name="__emb__",
                learning_rate=emb_lr,
                initializer=fluid.initializer.Xavier()),
            is_sparse=is_sparse)
        # vsum
        pt_sum = fluid.layers.sequence_pool(input=pt_emb, pool_type='sum')
        pt_ss = fluid.layers.softsign(pt_sum)
        # fc layer
        pt_fc = fluid.layers.fc(input=pt_ss,
                                size=hid_dim,
                                param_attr=fluid.ParamAttr(
                                    name="__fc__",
                                    learning_rate=base_lr,
                                    initializer=fluid.initializer.Xavier()),
                                bias_attr=fluid.ParamAttr(
                                    name="__fc_b__",
                                    initializer=fluid.initializer.Xavier()))

        # embedding
        nt_emb = fluid.layers.embedding(
            input=nt,
            size=[dict_dim, emb_dim],
            param_attr=fluid.ParamAttr(
                name="__emb__",
                learning_rate=emb_lr,
                initializer=fluid.initializer.Xavier()),
            is_sparse=is_sparse)

        # vsum
        nt_sum = fluid.layers.sequence_pool(input=nt_emb, pool_type='sum')
        nt_ss = fluid.layers.softsign(nt_sum)
        # fc layer
        nt_fc = fluid.layers.fc(input=nt_ss,
                                size=hid_dim,
                                param_attr=fluid.ParamAttr(
                                    name="__fc__",
                                    learning_rate=base_lr,
                                    initializer=fluid.initializer.Xavier()),
                                bias_attr=fluid.ParamAttr(
                                    name="__fc_b__",
                                    initializer=fluid.initializer.Xavier()))
        cos_q_pt = fluid.layers.cos_sim(q_fc, pt_fc)
        cos_q_nt = fluid.layers.cos_sim(q_fc, nt_fc)
        # loss
        avg_cost = self.get_loss(cos_q_pt, cos_q_nt, params)

    def get_loss(self, cos_q_pt, cos_q_nt):
        loss_op1 = fluid.layers.elementwise_sub(
            fluid.layers.fill_constant_batch_size_like(
                input=cos_q_pt,
                shape=[-1, 1],
                value=params.margin,
                dtype='float32'),
            cos_q_pt)
        loss_op2 = fluid.layers.elementwise_add(loss_op1, cos_q_nt)
        loss_op3 = fluid.layers.elementwise_max(
            fluid.layers.fill_constant_batch_size_like(
                input=loss_op2, shape=[-1, 1], value=0.0, dtype='float32'),
            loss_op2)
        avg_cost = fluid.layers.mean(loss_op3)
        return avg_cost

    def optimizer(self):
        optimizer = paddle.optimizer.SGD(self.learning_rate)
        return optimizer

    def infer_net(self):
        pass
