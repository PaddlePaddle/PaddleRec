# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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


class DPINLayer(nn.Layer):
    def __init__(self, K, emb_dim, max_item, max_context, d_model, h,
                 is_sparse):
        super(DPINLayer, self).__init__()

        self.emb_dim = emb_dim
        self.is_sparse = is_sparse
        self.max_item = max_item
        self.max_context = max_context
        self.K = K
        self.d_model = d_model
        self.h = h

        # Base Module
        # User Feature Embedding
        self.user_feat_emb = nn.Embedding(
            self.max_item,
            self.emb_dim,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=nn.initializer.XavierUniform()),
            name="user_feat_emb")
        # Context Feature Embedding
        self.context_feat_emb = nn.Embedding(
            self.max_context,
            self.emb_dim,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=nn.initializer.XavierUniform()),
            name="context_feat_emb")
        # Item Feature Embedding
        self.item_feat_emb = nn.Embedding(
            self.max_item,
            self.emb_dim,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=nn.initializer.XavierUniform()),
            name="item_feat_emb")
        self.base_module = nn.Sequential(
            nn.Linear(
                in_features=2 * self.emb_dim,
                out_features=1024,
                weight_attr=nn.initializer.KaimingUniform()),
            nn.ReLU(),
            nn.Linear(
                in_features=1024,
                out_features=512,
                weight_attr=nn.initializer.KaimingUniform()),
            nn.ReLU(),
            nn.Linear(
                in_features=512,
                out_features=128,
                weight_attr=nn.initializer.KaimingUniform()),
            nn.ReLU())

        # Deep Position-wise Interaction Module
        # Position Embedding
        self.position_emb = nn.Embedding(
            self.K,
            self.emb_dim,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=nn.initializer.XavierUniform()),
            name="position_emb")
        # Position-wise Interest Aggregation
        self.interest_agg = InterestAggregation(
            K, emb_dim, max_item, max_context, d_model, h, is_sparse)
        # Position-wise Non-linear Interaction
        self.non_linear_interaction = nn.Sequential(
            nn.Linear(
                in_features=3 * self.emb_dim,
                out_features=64,
                weight_attr=nn.initializer.KaimingUniform()),
            nn.ReLU())
        # Transformer Block
        self.transformer = nn.Sequential(
            Transformer(self.d_model, self.h, self.K),
            Transformer(self.d_model, self.h, self.K),
            Transformer(self.d_model, self.h, self.K),
            Transformer(self.d_model, self.h, self.K),
            Transformer(self.d_model, self.h, self.K),
            Transformer(self.d_model, self.h, self.K),
            Transformer(self.d_model, self.h, self.K),
            Transformer(self.d_model, self.h, self.K),
            Transformer(self.d_model, self.h, self.K),
            Transformer(self.d_model, self.h, self.K),
            Transformer(self.d_model, self.h, self.K),
            Transformer(self.d_model, self.h, self.K))

        # Position-wise Combination Module
        self.combination = nn.Sequential(
            nn.Linear(
                in_features=128 + self.d_model + self.emb_dim,
                out_features=128,
                weight_attr=nn.initializer.KaimingUniform()),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=1,
                weight_attr=nn.initializer.KaimingUniform()),
            nn.Sigmoid())
        self.position_emb_2 = nn.Embedding(
            self.K,
            self.emb_dim,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=nn.initializer.XavierUniform()),
            name="position_emb")

    def forward(self, hist_item, hist_cat, target_item, target_cat, position):
        # Base Module
        # Input: target_item, target_cat
        # Output: [batchsize, 128] as input of Position-wise Combination Module
        context_feat_emb_val = self.context_feat_emb(
            target_cat)  # [*, emb_dim]
        item_feat_emb_val = self.item_feat_emb(target_item)  # [*, emb_dim]
        base_module_input = paddle.concat(
            [context_feat_emb_val, item_feat_emb_val],
            axis=1)  # [*, 2*emb_dim]
        item_output = self.base_module(base_module_input)  # [batchsize, 128]

        # Deep Position-wise Interaction Module
        interest_agg = self.interest_agg(hist_item, hist_cat)  # [*, K, 2E]
        position_emb_val = self.position_emb(position)  # [*, K, E]
        # input of transformer block
        # ReLU(Conncat(E(k),c,b)W + b)
        input_non_linear_inter = paddle.concat(
            [position_emb_val, interest_agg], axis=2)
        non_linear_inter_val = self.non_linear_interaction(
            input_non_linear_inter)

        # Transformer Block
        transformer_output = self.transformer(
            non_linear_inter_val)  # [batchsize, 25, 64]

        # Position-wise Combination Module
        item_output_unqueeze = paddle.unsqueeze(item_output, axis=1)
        item_output_unqueeze = paddle.tile(
            item_output_unqueeze, repeat_times=[1, self.K, 1])
        com_position_emb_val = self.position_emb_2(position)
        item_pos = paddle.concat(
            [item_output_unqueeze, transformer_output, com_position_emb_val],
            axis=2)
        output = self.combination(item_pos)

        return output


class Transformer(nn.Layer):
    def __init__(self, d_model, h, K):
        super(Transformer, self).__init__()

        # d_model=64,h=2,K=25
        self.multi_head_attn = nn.MultiHeadAttention(d_model, h)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=[K, d_model])
        self.feed_forward = nn.Sequential(
            nn.Linear(
                in_features=d_model,
                out_features=d_model,
                weight_attr=nn.initializer.KaimingUniform()),
            nn.ReLU(),
            nn.Linear(
                in_features=d_model,
                out_features=d_model,
                weight_attr=nn.initializer.KaimingUniform()), )
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=[K, d_model])

    def forward(self, non_linear_inter_val):
        # MultiHeadAttention
        multi_head_output = self.multi_head_attn(
            non_linear_inter_val, non_linear_inter_val, non_linear_inter_val)
        # Add & Norm
        add_1 = paddle.add(non_linear_inter_val, multi_head_output)
        norm_1 = self.layer_norm_1(add_1)
        # Feed Forward
        feed = self.feed_forward(norm_1)
        # Add & Norm
        add_2 = paddle.add(norm_1, feed)
        norm_2 = self.layer_norm_1(add_2)
        return norm_2


class InterestAggregation(nn.Layer):
    def __init__(self, K, emb_dim, max_item, max_context, d_model, h,
                 is_sparse):
        super(InterestAggregation, self).__init__()
        self.emb_dim = emb_dim
        self.is_sparse = is_sparse
        self.max_item = max_item
        self.max_context = max_context
        self.K = K
        self.d_model = d_model
        self.h = h

        # User Beahvior Item Embedding
        self.user_bx_item_emb = nn.Embedding(
            self.max_item,
            self.emb_dim,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=nn.initializer.XavierUniform()),
            name="user_bx_item_emb")
        # User Beahvior Context Embedding
        self.user_bx_context_emb = nn.Embedding(
            self.max_context,
            self.emb_dim,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=nn.initializer.XavierUniform()),
            name="user_bx_context_emb")
        # 
        self.MLP = nn.Sequential(
            nn.Linear(
                in_features=2 * self.emb_dim,
                out_features=2 * self.emb_dim,
                weight_attr=nn.initializer.KaimingUniform()),
            nn.ReLU(),
            nn.Linear(
                in_features=2 * self.emb_dim,
                out_features=2 * self.emb_dim,
                weight_attr=nn.initializer.KaimingUniform()))

    def forward(self, hist_item, hist_cat):
        user_bx_item_val = self.user_bx_item_emb(
            hist_item)  # [*, K, L] => [*, K, L, E]
        user_bx_context_val = self.user_bx_context_emb(
            hist_cat)  # [*, K, L] => [*, K, L, E]
        user_bx = paddle.concat(
            [user_bx_item_val, user_bx_context_val], axis=3)  # [*, K, L, 2E]
        user_bx_exp = paddle.exp(self.MLP(user_bx))  # [*, K, L, 2E]
        user_bx_exp_sum = paddle.sum(user_bx_exp, axis=2)  # [*, K, 2E]
        # user_bx * user_bx_exp -> sum -> A / user_bx_exp_sum
        output = paddle.sum(user_bx * user_bx_exp,
                            axis=2) / user_bx_exp_sum  # [*, K, 2E]

        return output
