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

default_type = paddle.get_default_dtype()


class ENSFMLayer(nn.Layer):
    def __init__(self, num_users, num_items, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_users = num_users
        self.num_items = num_items

        self.user_feature_emb = nn.Embedding(
            self.num_users,
            self.embedding_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.01)))
        self.H_i = paddle.create_parameter(
            [self.embedding_size, 1],
            default_type,
            default_initializer=nn.initializer.Constant(0.01))
        self.H_s = paddle.create_parameter(
            [self.embedding_size, 1],
            default_type,
            default_initializer=nn.initializer.Constant(0.01))
        self.all_item_feature_emb = nn.Embedding(
            self.num_items + 1,
            self.embedding_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.01)))

        self.user_bias = nn.Embedding(
            self.num_users,
            1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.01)))
        self.item_bias = nn.Embedding(
            self.num_items,
            1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.01)))
        self.bias = paddle.create_parameter(
            [1], default_type, default_initializer=nn.initializer.Constant(0.))

    def forward(self, input_u, item_attribute, input_ur=None,
                item_bind_M=None):
        user_feature_emb = self.user_feature_emb(input_u)
        summed_user_emb = user_feature_emb.sum(1)
        all_item_feature_emb = self.all_item_feature_emb(item_attribute)
        summed_all_item_emb = all_item_feature_emb.sum(1)
        user_cross = 0.5 * (summed_user_emb**2 - (user_feature_emb**2).sum(1))
        item_cross = 0.5 * (summed_all_item_emb**2 -
                            (all_item_feature_emb**2).sum(1))
        user_cross_score = user_cross.matmul(self.H_s)
        item_cross_score = item_cross.matmul(self.H_s)
        user_bias = self.user_bias(input_u).sum(1)
        item_bias = self.item_bias(item_attribute).sum(1)

        I = paddle.ones([input_u.shape[0], 1])
        p_emb = paddle.concat(
            [summed_user_emb, user_cross_score + user_bias + self.bias, I], 1)

        I = paddle.ones([summed_all_item_emb.shape[0], 1])
        q_emb = paddle.concat(
            [summed_all_item_emb, I, item_cross_score + item_bias], 1)

        H_i_emb = paddle.concat([self.H_i, paddle.ones([2, 1])], 0)
        dot = p_emb.unsqueeze(1) * q_emb.unsqueeze(0)
        pre = paddle.matmul(dot, H_i_emb.unsqueeze(0)).sum(-1)

        # dot = paddle.einsum('ac,bc->abc', p_emb, q_emb)
        # pre = paddle.einsum('ajk,kl->aj', dot, H_i_emb)
        if input_ur is None:
            return (pre, )

        pos_item = F.embedding(input_ur, q_emb)
        pos_num_r = (input_ur != item_bind_M).astype(default_type)
        pos_item = paddle.einsum('ab,abc->abc', pos_num_r, pos_item)

        pos_r = paddle.einsum('ac,abc->abc', p_emb, pos_item)
        pos_r = paddle.einsum('ajk,kl->ajl', pos_r, H_i_emb).reshape(
            [input_u.shape[0], -1])
        return pre, pos_r, q_emb, p_emb, H_i_emb
