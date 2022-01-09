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

import sys

FLOAT_MIN = -sys.float_info.max


class PointWiseFeedForward(paddle.nn.Layer):
    def __init__(self, hidden_units, dropout_rate):  # wried, why fusion X 2?

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = paddle.nn.Conv1D(
            hidden_units,
            hidden_units,
            kernel_size=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.dropout1 = paddle.nn.Dropout(p=dropout_rate)
        self.relu = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv1D(
            hidden_units,
            hidden_units,
            kernel_size=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.dropout2 = paddle.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(self.conv1(inputs.transpose([0, 2, 1]))))))
        outputs = outputs.transpose(
            [0, 2, 1])  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class TimeAwareMultiHeadAttention(paddle.nn.Layer):
    def __init__(self, hidden_size, head_num, dropout_rate):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = paddle.nn.Linear(
            hidden_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.K_w = paddle.nn.Linear(
            hidden_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.V_w = paddle.nn.Linear(
            hidden_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))

        self.dropout = paddle.nn.Dropout(p=dropout_rate)
        self.softmax = paddle.nn.Softmax(-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K,
                time_matrix_V, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        if self.head_num > 1:

            Q_ = paddle.concat(paddle.split(Q, self.head_num, axis=2), axis=0)
            K_ = paddle.concat(paddle.split(K, self.head_num, axis=2), axis=0)
            V_ = paddle.concat(paddle.split(V, self.head_num, axis=2), axis=0)

            time_matrix_K_ = paddle.concat(
                paddle.split(
                    time_matrix_K, self.head_num, axis=3), axis=0)
            time_matrix_V_ = paddle.concat(
                paddle.split(
                    time_matrix_V, self.head_num, axis=3), axis=0)
            abs_pos_K_ = paddle.concat(
                paddle.split(
                    abs_pos_K, self.head_num, axis=2), axis=0)
            abs_pos_V_ = paddle.concat(
                paddle.split(
                    abs_pos_V, self.head_num, axis=2), axis=0)
        else:
            Q_ = Q
            K_ = K
            V_ = V
            time_matrix_K_ = time_matrix_K
            time_matrix_V_ = time_matrix_V
            abs_pos_K_ = abs_pos_K
            abs_pos_V_ = abs_pos_V

        # print(Q_.shape, time_matrix_K_.shape, abs_pos_K_.shape)

        # batched channel wise matmul to gen attention weights
        attn_weights = paddle.matmul(Q_, K_, transpose_y=True)
        attn_weights += paddle.matmul(Q_, abs_pos_K_, transpose_y=True)
        # print(time_matrix_K_.shape, Q_.shape)
        attn_weights += paddle.matmul(time_matrix_K_,
                                      Q_.unsqueeze(-1)).squeeze(-1)
        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1]**0.5)

        # key masking, -2^32 lead to leaking, inf lead to nan
        # 0 * inf = nan, then reduce_sum([nan,...]) = nan

        # time_mask = time_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        time_mask = time_mask.astype('int32').unsqueeze(-1).tile(
            [self.head_num, 1, 1])
        time_mask = time_mask.expand(
            [-1, -1, attn_weights.shape[-1]]).astype(paddle.bool)
        # print(attn_mask.shape)
        attn_mask = attn_mask.astype('int32').unsqueeze(0).expand(
            [attn_weights.shape[0], -1, -1]).astype(paddle.bool)
        # print(attn_mask.shape)
        # paddings = paddle.ones(attn_weights.shape) * (-2 ** 32 + 1)  # -1e23 # float('-inf')
        paddings = paddle.ones_like(attn_weights) * (
            -2**32 + 1)  # -1e23 # float('-inf')
        # print(attn_mask.shape, paddings.shape, attn_weights.shape)
        attn_weights = paddle.where(time_mask, paddings,
                                    attn_weights)  # True:pick padding
        attn_weights = paddle.where(attn_mask, paddings,
                                    attn_weights)  # enforcing causality

        attn_weights = self.softmax(attn_weights)
        # attn_weights = paddle.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(
            outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        if self.head_num > 1:
            outputs = paddle.concat(
                paddle.split(
                    outputs, self.head_num, axis=0), axis=2)  # div batch_size
        return outputs


class TiSASRecLayer(paddle.nn.Layer):
    def __init__(self,
                 user_num,
                 item_num,
                 hidden_units,
                 maxlen,
                 time_span,
                 num_blocks,
                 num_heads,
                 dropout_rate=0.2):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num

        # TODO: loss += l2_emb for regularizing embedding vectors during training
        self.item_emb = paddle.nn.Embedding(
            self.item_num + 1,
            hidden_units,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.item_emb_dropout = paddle.nn.Dropout(p=dropout_rate)

        self.abs_pos_K_emb = paddle.nn.Embedding(
            maxlen,
            hidden_units,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.abs_pos_V_emb = paddle.nn.Embedding(
            maxlen,
            hidden_units,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.time_matrix_K_emb = paddle.nn.Embedding(
            time_span + 1,
            hidden_units,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.time_matrix_V_emb = paddle.nn.Embedding(
            time_span + 1,
            hidden_units,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))

        self.item_emb_dropout = paddle.nn.Dropout(p=dropout_rate)
        self.abs_pos_K_emb_dropout = paddle.nn.Dropout(p=dropout_rate)
        self.abs_pos_V_emb_dropout = paddle.nn.Dropout(p=dropout_rate)
        self.time_matrix_K_dropout = paddle.nn.Dropout(p=dropout_rate)
        self.time_matrix_V_dropout = paddle.nn.Dropout(p=dropout_rate)

        self.attention_layernorms = paddle.nn.LayerList(
        )  # to be Q for self-attention
        self.attention_layers = paddle.nn.LayerList()
        self.forward_layernorms = paddle.nn.LayerList()
        self.forward_layers = paddle.nn.LayerList()

        self.last_layernorm = paddle.nn.LayerNorm(hidden_units, epsilon=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = paddle.nn.LayerNorm(
                hidden_units, epsilon=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(
                hidden_units, num_heads, dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = paddle.nn.LayerNorm(hidden_units, epsilon=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = paddle.nn.Sigmoid()
            # self.neg_sigmoid = paddle.nn.Sigmoid()

    def seq2feats(self, log_seqs, time_matrices):
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb._embedding_dim**0.5
        seqs = self.item_emb_dropout(seqs)
        positions = paddle.arange(log_seqs.shape[1]).unsqueeze(0).expand(
            [log_seqs.shape[0], -1])
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # mask 0th items(placeholder for dry-run) in log_seqs
        # would be easier if 0th item could be an exception for training
        timeline_mask = log_seqs == 0
        seqs *= (log_seqs != 0).astype(paddle.get_default_dtype()).unsqueeze(
            -1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = (
            paddle.tril(paddle.ones([tl, tl])) == 0).astype(paddle.bool)

        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            # seqs = paddle.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](
                Q, seqs, timeline_mask, attention_mask, time_matrix_K,
                time_matrix_V, abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs
            # seqs = paddle.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

            seqs *= (timeline_mask.astype(int) == 0
                     ).astype(paddle.get_default_dtype()).unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self,
                log_seqs,
                time_matrices,
                item_indices=None,
                pos_seqs=None,
                neg_seqs=None):
        log_feats = self.seq2feats(log_seqs, time_matrices)
        if item_indices is not None:
            final_feat = log_feats[:,
                                   -1, :]  # only use last QKV classifier, a waste
            item_embs = self.item_emb(item_indices[0])  # (U, I, C)
            logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
            return logits  # preds # (U, I)

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)
        pos_logits = (log_feats * pos_embs).sum(-1)
        neg_logits = (log_feats * neg_embs).sum(-1)
        return pos_logits, neg_logits  # pos_pred, neg_pred
