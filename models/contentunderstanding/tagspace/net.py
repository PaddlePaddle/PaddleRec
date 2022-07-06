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


class TagspaceLayer(nn.Layer):
    def __init__(self, vocab_text_size, vocab_tag_size, emb_dim, hid_dim,
                 win_size, margin, neg_size, text_len):
        super(TagspaceLayer, self).__init__()
        self.vocab_text_size = vocab_text_size
        self.vocab_tag_size = vocab_tag_size
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.win_size = win_size
        self.margin = margin
        self.neg_size = neg_size
        self.text_len = text_len

        self.text_embedding = paddle.nn.Embedding(
            self.vocab_text_size,
            self.emb_dim,
            padding_idx=75377,
            sparse=True,
            name="text_emb")
        self.tag_embedding = paddle.nn.Embedding(
            self.vocab_tag_size, self.emb_dim, sparse=True, name="tag_emb")

        self.conv = nn.Conv1D(
            in_channels=self.emb_dim,
            out_channels=self.hid_dim,
            kernel_size=self.win_size,
            data_format='NLC')

        self.hid_fc = paddle.nn.Linear(
            in_features=self.hid_dim,
            out_features=self.emb_dim,
            name="text_hid")

    def forward(self, inputs):
        text = inputs[0]
        pos_tag = inputs[1]
        neg_tag = inputs[2]

        text_emb = self.text_embedding(text)
        text_emb = paddle.reshape(
            text_emb, shape=[-1, self.text_len, self.emb_dim])
        pos_tag_emb = self.tag_embedding(pos_tag)
        pos_tag_emb = paddle.reshape(pos_tag_emb, shape=[-1, self.emb_dim])
        neg_tag_emb = self.tag_embedding(neg_tag)
        neg_tag_emb = paddle.reshape(
            neg_tag_emb, shape=[-1, self.neg_size, self.emb_dim])

        conv_1d = self.conv(text_emb)
        act = paddle.tanh(conv_1d)
        maxpool = paddle.max(act, axis=1)
        maxpool = paddle.reshape(maxpool, shape=[-1, self.hid_dim])
        text_hid = self.hid_fc(maxpool)
        cos_pos = F.cosine_similarity(
            pos_tag_emb, text_hid, axis=1).reshape([-1, 1])
        neg_tag_emb = paddle.max(neg_tag_emb, axis=1)
        neg_tag_emb = paddle.reshape(neg_tag_emb, shape=[-1, self.emb_dim])
        cos_neg = F.cosine_similarity(
            neg_tag_emb, text_hid, axis=1).reshape([-1, 1])
        return cos_pos, cos_neg
