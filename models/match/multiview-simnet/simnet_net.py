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
import math
import numpy as np
import os
import paddle.fluid as fluid


class MultiviewSimnetLayer(nn.Layer):
    def __init__(self, query_encoder, title_encoder, query_encode_dim,
                 title_encode_dim, emb_size, emb_dim, hidden_size, margin,
                 batch_size):
        super(MultiviewSimnetLayer, self).__init__()
        self.query_encoder = query_encoder
        self.title_encoder = title_encoder
        self.query_encode_dim = query_encode_dim
        self.title_encode_dim = title_encode_dim
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.margin = margin
        self.batch_size = batch_size

        self.embedding = paddle.nn.Embedding(
            self.emb_size, self.emb_dim, padding_idx=0, name="emb")

        # grnn-encoder
        if self.query_encoder == "gru":
            self.gru = paddle.nn.GRU(self.emb_dim,
                                     self.query_encode_dim,
                                     bias_ih_attr=False)
        # cnn-encoder
        self.cnn_encoder = []
        if self.query_encoder == "cnn":
            self.conv = nn.Conv1D(
                in_channels=self.emb_dim,
                out_channels=128,
                kernel_size=3,
                data_format='NLC')
            self.cnn_encoder.append(self.conv)
            self.act = paddle.nn.ReLU()
            self.cnn_encoder.append(self.act)

        self.q_fc = paddle.nn.Linear(
            in_features=128, out_features=self.hidden_size, name="q_fc")
        self.t_fc = paddle.nn.Linear(
            in_features=128, out_features=self.hidden_size, name="t_fc")

    def forward(self, inputs, is_infer=False):
        self.q_slots = inputs[0]
        self.pt_slots = inputs[1]
        if not is_infer:
            self.nt_slots = inputs[2]

        q_embs = [self.embedding(query) for query in self.q_slots]
        q_encodes = []
        for emb in q_embs:
            emb = paddle.reshape(
                emb, shape=[self.batch_size, -1, self.query_encode_dim])
            gru = self.gru(emb)
            maxpool = paddle.max(gru[0], axis=1)
            maxpool = paddle.reshape(
                maxpool, shape=[self.batch_size, self.query_encode_dim])
            q_encodes.append(maxpool)
        q_concat = paddle.concat(q_encodes, axis=1)
        q_hid = self.q_fc(q_concat)

        pt_embs = [self.embedding(title) for title in self.pt_slots]
        pt_encodes = []
        for emb in pt_embs:
            emb = paddle.reshape(
                emb, shape=[self.batch_size, -1, self.title_encode_dim])
            gru = self.gru(emb)
            maxpool = paddle.max(gru[0], axis=1)
            maxpool = paddle.reshape(
                maxpool, shape=[self.batch_size, self.title_encode_dim])
            pt_encodes.append(maxpool)
        pt_concat = paddle.concat(pt_encodes, axis=1)
        pt_hid = self.t_fc(pt_concat)

        cos_pos = F.cosine_similarity(q_hid, pt_hid, axis=1).reshape([-1, 1])
        if is_infer:
            return cos_pos, paddle.ones(shape=[self.batch_size, 1])

        nt_embs = [self.embedding(title) for title in self.nt_slots]
        nt_encodes = []
        for emb in nt_embs:
            emb = paddle.reshape(
                emb, shape=[self.batch_size, -1, self.title_encode_dim])
            gru = self.gru(emb)
            maxpool = paddle.max(gru[0], axis=1)
            maxpool = paddle.reshape(
                maxpool, shape=[self.batch_size, self.title_encode_dim])
            nt_encodes.append(maxpool)
        nt_concat = paddle.concat(nt_encodes, axis=1)
        nt_hid = self.t_fc(nt_concat)

        cos_neg = F.cosine_similarity(q_hid, nt_hid, axis=1).reshape([-1, 1])
        return cos_pos, cos_neg
