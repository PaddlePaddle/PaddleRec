# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.nn import Conv1D
import paddle
import paddle.nn as nn
import math
import numpy as np


class DINLayer(nn.Layer):
    def __init__(self, item_emb_size, cat_emb_size, act, is_sparse,
                 use_DataLoader, item_count, cat_count):
        super(DINLayer, self).__init__()

        self.item_emb_size = item_emb_size
        self.cat_emb_size = cat_emb_size
        self.act = act
        self.is_sparse = is_sparse
        self.use_DataLoader = use_DataLoader
        self.item_count = item_count
        self.cat_count = cat_count

        self.hist_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            name="item_emb")
        self.hist_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            name="cat_emb")
        self.target_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            name="item_emb")
        self.target_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            name="cat_emb")
        self.target_item_seq_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            name="item_emb")

        self.target_cat_seq_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            name="cat_emb")

        self.item_b_attr = paddle.nn.Embedding(
            self.item_count,
            1,
            sparse=self.is_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))

        self.attention_layer = []
        sizes = [(self.item_emb_size + self.cat_emb_size) * 4
                 ] + [80] + [40] + [1]
        acts = ["sigmoid" for _ in range(len(sizes) - 2)] + [None]

        for i in range(len(sizes) - 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0)))
            self.add_sublayer('linear_%d' % i, linear)
            self.attention_layer.append(linear)
            if acts[i] == 'sigmoid':
                act = paddle.nn.Sigmoid()
                self.add_sublayer('act_%d' % i, act)
                self.attention_layer.append(act)

        self.con_layer = []

        self.firInDim = self.item_emb_size + self.cat_emb_size
        self.firOutDim = self.item_emb_size + self.cat_emb_size

        linearCon = paddle.nn.Linear(
            in_features=self.firInDim,
            out_features=self.firOutDim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))
        self.add_sublayer('linearCon', linearCon)
        self.con_layer.append(linearCon)

        conDim = self.item_emb_size + self.cat_emb_size + self.item_emb_size + self.cat_emb_size

        conSizes = [conDim] + [80] + [40] + [1]
        conActs = ["sigmoid" for _ in range(len(conSizes) - 2)] + [None]

        for i in range(len(conSizes) - 1):
            linear = paddle.nn.Linear(
                in_features=conSizes[i],
                out_features=conSizes[i + 1],
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0)))
            self.add_sublayer('linear_%d' % i, linear)
            self.con_layer.append(linear)
            if conActs[i] == 'sigmoid':
                act = paddle.nn.Sigmoid()
                self.add_sublayer('act_%d' % i, act)
                self.con_layer.append(act)

    def forward(self, hist_item_seq, hist_cat_seq, target_item, target_cat,
                label, mask, target_item_seq, target_cat_seq):
        hist_item_emb = self.hist_item_emb_attr(hist_item_seq)
        hist_cat_emb = self.hist_cat_emb_attr(hist_cat_seq)
        target_item_emb = self.target_item_emb_attr(target_item)
        target_cat_emb = self.target_cat_emb_attr(target_cat)
        target_item_seq_emb = self.target_item_seq_emb_attr(target_item_seq)
        target_cat_seq_emb = self.target_cat_seq_emb_attr(target_cat_seq)
        item_b = self.item_b_attr(target_item)

        hist_seq_concat = paddle.concat([hist_item_emb, hist_cat_emb], axis=2)
        target_seq_concat = paddle.concat(
            [target_item_seq_emb, target_cat_seq_emb], axis=2)
        target_concat = paddle.concat(
            [target_item_emb, target_cat_emb], axis=1)

        concat = paddle.concat(
            [
                hist_seq_concat, target_seq_concat,
                hist_seq_concat - target_seq_concat,
                hist_seq_concat * target_seq_concat
            ],
            axis=2)

        for attlayer in self.attention_layer:
            concat = attlayer(concat)

        atten_fc3 = concat + mask.astype(concat.dtype)
        atten_fc3 = paddle.transpose(atten_fc3, perm=[0, 2, 1])
        atten_fc3 = paddle.scale(atten_fc3, scale=self.firInDim**-0.5)
        weight = paddle.nn.functional.softmax(atten_fc3)

        output = paddle.matmul(weight, hist_seq_concat)

        output = paddle.reshape(output, shape=[0, self.firInDim])

        for firLayer in self.con_layer[:1]:
            concat = firLayer(output)

        embedding_concat = paddle.concat([concat, target_concat], axis=1)

        for colayer in self.con_layer[1:]:
            embedding_concat = colayer(embedding_concat)

        logit = embedding_concat + item_b
        return logit
