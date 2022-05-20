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
from paddle import ParamAttr
import paddle.nn.functional as F
import numpy as np
import math

initializer = nn.initializer


class GateAttention(paddle.nn.Layer):
    '''gate attention to aggregate historical interacted packages'''

    def __init__(self, hidden_size):
        super().__init__()
        self.w1 = nn.Linear(
            hidden_size,
            hidden_size,
            weight_attr=ParamAttr(initializer=initializer.Normal(std=0.1)),
            bias_attr=False)
        self.w2 = nn.Linear(
            hidden_size,
            hidden_size,
            weight_attr=ParamAttr(initializer=initializer.Normal(std=0.1)), )

    def forward(self, inputs1, inputs2, mask):
        x1 = self.w1(inputs1).unsqueeze(1)
        x2 = self.w2(inputs2)
        attn = F.sigmoid(x1 + x2)
        attn = attn * mask
        return attn * inputs2


class Attention(paddle.nn.Layer):
    def __init__(self, hidden_size1, hidden_size2, dropout_rate):
        super().__init__()
        self.w_omega = nn.Linear(
            hidden_size2,
            1,
            weight_attr=ParamAttr(initializer=initializer.Normal(std=0.1)))
        self.w = nn.Linear(
            hidden_size1,
            hidden_size2,
            weight_attr=ParamAttr(initializer=initializer.Normal(std=0.1)),
            bias_attr=False)
        self.u_omega = paddle.create_parameter(
            [1], 'float32', default_initializer=initializer.Normal(std=0.1))

        self.bn = nn.BatchNorm1D(hidden_size2, data_format='NLC')
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, inputs1, inputs2, masks):
        inputs = paddle.concat([inputs1, inputs2], -1)

        inputs = self.dropout(F.relu(self.bn(self.w(inputs))))
        v = F.tanh(self.w_omega(inputs))
        vu = paddle.matmul(v, self.u_omega)
        paddings = paddle.ones_like(vu) * (-2**32 + 1)
        vu = paddle.where(masks.astype('bool'), paddings, vu)
        att = F.softmax(vu, -1)
        f_emb = (inputs2 * att.unsqueeze(-1)).sum(1)
        return f_emb, att


class IntraLayer(nn.Layer):
    def __init__(self, hidden_size, f_max_len):
        super(IntraLayer, self).__init__()
        self.f_max_len = f_max_len
        self.w_k = nn.Linear(
            hidden_size,
            hidden_size,
            weight_attr=ParamAttr(initializer=initializer.XavierNormal()),
            bias_attr=False)
        self.w_i = nn.Linear(
            hidden_size,
            hidden_size,
            weight_attr=ParamAttr(initializer=initializer.XavierNormal()),
            bias_attr=False)
        self.u_omega = paddle.create_parameter(
            [1], 'float32', default_initializer=initializer.Normal(std=0.1))
        self.w_omega = nn.Linear(
            2 * hidden_size,
            1,
            weight_attr=ParamAttr(initializer=initializer.Normal(std=0.1)), )

    def forward(self, item_emb, friend_emb, masks):
        f_k_emb = self.w_k(friend_emb)
        _item = self.w_i(item_emb).unsqueeze(1)
        inputs = paddle.concat(
            [paddle.tile(_item, [1, self.f_max_len, 1]), f_k_emb], -1)
        v = F.tanh(self.w_omega(inputs))

        vu = paddle.matmul(v, self.u_omega)
        paddings = paddle.ones_like(vu) * (-2**32 + 1)
        x = paddle.where(masks == 0, paddings, vu)
        att = F.softmax(x, -1)
        output = (f_k_emb * att.unsqueeze(-1)).sum(1, keepdim=True)
        return output


class IPRECLayer(paddle.nn.Layer):
    def __init__(self, user_num, item_num, biz_num, hidden_units, f_max_len, k,
                 u_max_i, u_max_f, u_max_pack, pack_max_nei_b, pack_max_nei_f,
                 dropout_rate):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.biz_num = biz_num
        self.hidden_units = hidden_units
        self.f_max_len = f_max_len
        self.u_max_i = u_max_i
        self.u_max_f = u_max_f
        self.u_max_pack = u_max_pack
        self.pack_max_nei_b = pack_max_nei_b
        self.pack_max_nei_f = pack_max_nei_f
        self.K = k

        self.item_emb = nn.Embedding(
            self.item_num + 1,
            hidden_units,
            padding_idx=0,
            weight_attr=ParamAttr(initializer=initializer.XavierUniform()))
        self.user_emb = nn.Embedding(
            self.user_num + 1,
            hidden_units,
            padding_idx=0,
            weight_attr=ParamAttr(initializer=initializer.XavierUniform()))
        self.biz_emb = nn.Embedding(
            self.biz_num + 1,
            hidden_units,
            padding_idx=0,
            weight_attr=ParamAttr(initializer=initializer.XavierUniform()))

        self.friend_attn = Attention(2 * hidden_units, hidden_units,
                                     dropout_rate)
        self.item_attn = Attention(2 * hidden_units, hidden_units,
                                   dropout_rate)
        self.biz_attn = Attention(2 * hidden_units, hidden_units, dropout_rate)
        self.type_attn = Attention(2 * hidden_units, hidden_units,
                                   dropout_rate)
        self.pack_attn = Attention(2 * hidden_units, hidden_units,
                                   dropout_rate)
        self.w_self = nn.Linear(
            2 * hidden_units,
            hidden_units,
            weight_attr=ParamAttr(initializer=initializer.XavierUniform()),
            bias_attr=False)

        self.intra_layers = nn.LayerList(
            [IntraLayer(hidden_units, f_max_len) for _ in range(self.K)])
        self.dropout = nn.Dropout(dropout_rate)

        self.w_a = nn.Linear(
            2 * hidden_units,
            hidden_units,
            weight_attr=ParamAttr(initializer=initializer.XavierUniform()),
            bias_attr=False)
        self.u_omega = paddle.create_parameter(
            [1], 'float32', default_initializer=initializer.Normal(std=0.1))
        self.w_omega = nn.Linear(
            hidden_units,
            1,
            weight_attr=ParamAttr(initializer=initializer.Normal(std=0.1)), )

        self.friend_gate = GateAttention(hidden_units)
        self.user_gate = GateAttention(hidden_units)
        self.biz_gate = GateAttention(hidden_units)

        self.dnn_size = [3 * hidden_units, 64, 32, 1]
        self.dnn = []
        for i in range(len(self.dnn_size) - 1):
            self.dnn.append(nn.Linear(self.dnn_size[i], self.dnn_size[i + 1]))
            if i != len(self.dnn_size) - 2:
                self.dnn.append(nn.ReLU())
                self.dnn.append(nn.Dropout(dropout_rate))
        self.dnn = nn.Sequential(*self.dnn)

    def forward(self, user, item, biz, friends, user_items, user_bizs,
                user_friends, user_packages, pack_neighbors_b,
                pack_neighbors_f):

        batch_size = user.shape[0]
        user = user.squeeze(1)
        item = item.squeeze(1)
        biz = biz.squeeze(1)
        user_emb = self.user_emb(user)
        up_mask = user_packages.sum(-1).abs().astype('float32').sign(
        ).unsqueeze(-1)
        pb_mask = pack_neighbors_b.sum(-1).abs().astype('float32').sign(
        ).unsqueeze(-1)
        pf_mask = pack_neighbors_f.sum(-1).abs().astype('float32').sign(
        ).unsqueeze(-1)
        up_items, up_bizs, up_friends = paddle.split(
            user_packages, [1, 1, self.f_max_len],
            axis=-1)  # B * u_max_p * 1, B * u_max_p * 1, B * u_max_p * max_f
        pb_items, pb_bizs, pb_friends = paddle.split(
            pack_neighbors_b, [1, 1, self.f_max_len],
            axis=-1)  # B * p_max_nei(biz) * 1
        pf_items, pf_bizs, pf_friends = paddle.split(
            pack_neighbors_f, [1, 1, self.f_max_len],
            axis=-1)  # B * p_max_nei(fri) * 1
        _items = paddle.concat(
            [paddle.reshape(item, [-1, 1, 1]), up_items, pb_items, pf_items],
            axis=1)
        _bizs = paddle.concat(
            [paddle.reshape(biz, [-1, 1, 1]), up_bizs, pb_bizs, pf_bizs],
            axis=1)
        _friends = paddle.concat(
            [friends.unsqueeze(1), up_friends, pb_friends, pf_friends], axis=1)
        user_emb, a2, a3, ta = self.dual_aggregate(user_emb, user_items,
                                                   user_bizs, user_friends)

        intra_packages, att = self.intra(
            user_emb,
            paddle.reshape(_friends, [-1, self.f_max_len]),
            paddle.reshape(_items, [-1]), paddle.reshape(_bizs, [-1]))
        intra_packages = paddle.reshape(intra_packages,
                                        [batch_size, -1, self.hidden_units])
        att = paddle.reshape(att, [batch_size, -1, 7])
        tar_pack, u_packs, pb_packs, pf_packs = paddle.split(
            intra_packages,
            [1, self.u_max_pack, self.pack_max_nei_b, self.pack_max_nei_f],
            axis=1)
        tar_att, u_att, pb_att, pf_att = paddle.split(
            att,
            [1, self.u_max_pack, self.pack_max_nei_b, self.pack_max_nei_f],
            axis=1)
        tar_pack = paddle.reshape(tar_pack, [batch_size, self.hidden_units])
        tar_att = paddle.reshape(tar_att, [batch_size, 7])
        u_packs = paddle.reshape(u_packs, [batch_size, -1, self.hidden_units])
        pb_packs = paddle.reshape(pb_packs,
                                  [batch_size, -1, self.hidden_units])
        pf_packs = paddle.reshape(pf_packs,
                                  [batch_size, -1, self.hidden_units])

        # gate_attention
        pack_emb = tar_pack + self.biz_gate(tar_pack, pb_packs, pb_mask).sum(1) + \
                   self.friend_gate(tar_pack, pf_packs, pf_mask).sum(1)

        # gate_attention
        user_emb = user_emb + self.user_gate(user_emb, u_packs, up_mask).sum(1)

        agg_out = self.dnn(
            paddle.concat([user_emb, pack_emb, user_emb * pack_emb], 1))

        scores_normalized = F.sigmoid(agg_out)
        return scores_normalized

    def intra(self, user_emb, friends, item, biz):
        _user_emb = paddle.reshape(
            paddle.tile(
                user_emb.unsqueeze(1), [
                    1, 1 + self.u_max_pack + self.pack_max_nei_b +
                    self.pack_max_nei_f, 1
                ]), [-1, self.hidden_units, 1])  # BN*D*1
        friend_emb = self.user_emb(friends)
        item_emb = self.item_emb(item)
        biz_emb = self.biz_emb(biz)
        masks = friend_emb.sum(-1).abs().astype('float32').sign()

        # social influence
        f_list = []
        for i in range(self.K):
            output = self.intra_layers[i](item_emb, friend_emb, masks)
            f_list.append(output)  # K*BN*D

        f_K_emb = paddle.concat(f_list, 1)

        t_user = paddle.reshape(
            paddle.tile(
                user_emb.unsqueeze(1), [
                    1, 1 + self.u_max_pack + self.pack_max_nei_b +
                    self.pack_max_nei_f, 1
                ]), [-1, 1, self.hidden_units])
        inputs = paddle.concat([paddle.tile(t_user, [1, self.K, 1]), f_K_emb],
                               -1)

        inputs = self.dropout(F.relu(self.w_a(inputs)))

        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (BN,F,D)*(D,A)=(BN,F,A), where A=attention_size
        v = F.tanh(self.w_omega(inputs))

        vu = paddle.matmul(v, self.u_omega)  # BN*F

        att = F.softmax(vu, -1)
        f_emb = (f_K_emb * att.unsqueeze(-1)).sum(1)

        # interaction
        pack = paddle.stack([
            f_emb, item_emb, biz_emb, f_emb * item_emb, f_emb * biz_emb,
            item_emb * biz_emb, f_emb * item_emb * biz_emb
        ], 1)
        masks = pack.sum(-1).abs().astype('float32').sign()
        _user_emb = paddle.tile(
            paddle.transpose(
                _user_emb, perm=[0, 2, 1]), [1, 7, 1])
        pack_emb, att = self.pack_attn(_user_emb, pack, masks)

        return pack_emb, att

    def dual_aggregate(self, user_emb, items, bizs, friends):
        '''dual_aggregate for modeling user feature'''
        friends_emb = self.user_emb(friends)  # B*M*D
        items_emb = self.item_emb(items)  # B*N*D
        bizs_emb = self.biz_emb(bizs)
        user_emb_ = user_emb.unsqueeze(1)

        f_masks = friends_emb.sum(-1).abs().astype('float32').sign()
        _user_emb = paddle.tile(user_emb_, [1, self.u_max_f, 1])
        friend_type, att1 = self.friend_attn(_user_emb, friends_emb, f_masks)

        i_masks = items_emb.sum(-1).abs().astype('float32').sign()
        _user_emb = paddle.tile(user_emb_, [1, self.u_max_i, 1])
        item_type, att2 = self.item_attn(_user_emb, items_emb, i_masks)

        b_masks = bizs_emb.sum(-1).abs().astype('float32').sign()
        _user_emb = paddle.tile(user_emb_, [1, self.u_max_i, 1])
        biz_type, att2 = self.biz_attn(_user_emb, bizs_emb, b_masks)

        inputs = paddle.stack([friend_type, item_type, biz_type], 1)
        masks = inputs.sum(-1).abs().astype('float32').sign()
        _user_emb = paddle.tile(user_emb_, [1, 3, 1])
        _user_emb, t_att = self.type_attn(_user_emb, inputs, masks)

        user_emb = F.relu(
            self.w_self(paddle.concat([_user_emb, user_emb], -1)))
        return user_emb, att1, att1, att1

