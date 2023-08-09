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
import paddle.nn.functional as F


class DIENLayer(nn.Layer):
    def __init__(self, item_emb_size, cat_emb_size, act, is_sparse,
                 use_DataLoader, item_count, cat_count):
        super(DIENLayer, self).__init__()
        self.epsilon = 0.00000001
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
            padding_idx=0,
            name="item_emb")
        self.hist_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="cat_emb")
        self.target_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="item_emb")
        self.target_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="cat_emb")
        self.target_item_seq_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="item_emb")

        self.target_cat_seq_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="cat_emb")

        self.neg_item_seq_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="item_emb")

        self.neg_cat_seq_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="cat_emb")

        self.item_b_attr = paddle.nn.Embedding(
            self.item_count,
            1,
            sparse=self.is_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))

        # ------------------------- attention net --------------------------
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

        # ------------------------- prev net --------------------------
        self.top_layer = []
        sizes = [(self.item_emb_size + self.cat_emb_size) * 2
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
            self.top_layer.append(linear)
            if acts[i] == 'sigmoid':
                act = paddle.nn.Sigmoid()
                self.add_sublayer('act_%d' % i, act)
                self.top_layer.append(act)

        # ------------------------- gru-net --------------------------

        self.gru_net = paddle.nn.GRU(
            input_size=self.item_emb_size + self.cat_emb_size,
            hidden_size=self.item_emb_size + self.cat_emb_size,
            num_layers=2)
        self.gru_cell_attention = paddle.nn.GRUCell(
            self.item_emb_size + self.cat_emb_size,
            self.item_emb_size + self.cat_emb_size)
        self.sigm = paddle.nn.Sigmoid()

        #------------------------- attention net --------------------------

    def forward(self, hist_item_seq, hist_cat_seq, target_item, target_cat,
                label, mask, target_item_seq, target_cat_seq,
                neg_hist_item_seq, neg_hist_cat_seq):
        # ------------------------- network data --------------------------
        hist_item_emb = self.hist_item_emb_attr(hist_item_seq)
        hist_cat_emb = self.hist_cat_emb_attr(hist_cat_seq)
        target_item_emb = self.target_item_emb_attr(target_item)
        target_cat_emb = self.target_cat_emb_attr(target_cat)
        target_item_seq_emb = self.target_item_seq_emb_attr(target_item_seq)
        target_cat_seq_emb = self.target_cat_seq_emb_attr(target_cat_seq)
        neg_hist_item_emb = self.neg_item_seq_emb_attr(neg_hist_item_seq)
        neg_hist_cat_emb = self.neg_cat_seq_emb_attr(neg_hist_cat_seq)
        item_b = self.item_b_attr(target_item)

        # ------------------------- Interest Extractor Layer --------------------------
        hist_seq_concat = paddle.concat([hist_item_emb, hist_cat_emb], axis=2)
        neg_hist_seq_concat = paddle.concat(
            [neg_hist_item_emb, neg_hist_cat_emb], axis=2)
        target_seq_concat = paddle.concat(
            [target_item_seq_emb, target_cat_seq_emb], axis=2)
        target_concat = paddle.concat(
            [target_item_emb, target_cat_emb], axis=1)
        gru_shape = hist_seq_concat.shape
        reshape_hist_item_emb = hist_seq_concat
        neg_reshape_hist_item_emb = neg_hist_seq_concat
        gru_hist_item_emb = hist_seq_concat
        gru_out, gru_hid = self.gru_net(gru_hist_item_emb)

        # ------------------------- attention --------------------------
        concat = paddle.concat(
            [
                hist_seq_concat, target_seq_concat,
                paddle.subtract(hist_seq_concat, target_seq_concat),
                paddle.multiply(hist_seq_concat, target_seq_concat)
            ],
            axis=2)
        for attlayer in self.attention_layer:
            concat = attlayer(concat)

        atten_fc3 = paddle.add(concat, mask)
        atten_fc3 = paddle.transpose(atten_fc3, perm=[0, 2, 1])
        atten_fc3 = paddle.scale(
            atten_fc3, scale=(self.item_emb_size + self.cat_emb_size)**-0.5)
        weight = paddle.nn.functional.softmax(atten_fc3)
        weighted = paddle.transpose(x=weight, perm=[0, 2, 1])
        weighted_vector = paddle.multiply(weighted, hist_seq_concat)
        weighted_vector = paddle.transpose(weighted_vector, perm=[1, 0, 2])
        # ------------------------- rnn-gru --------------------------
        concat_weighted_vector = paddle.concat([weighted_vector], axis=2)
        # ------------------------- Auxiliary loss  --------------------------
        start_value = paddle.zeros(shape=[1], dtype="float32")
        gru_out_pad = gru_out
        pos_seq_pad = reshape_hist_item_emb
        neg_seq_pad = neg_reshape_hist_item_emb
        INT_MAX = int(1.0 * 1e9)
        slice_gru = paddle.slice(
            gru_out_pad,
            axes=[0, 1, 2],
            starts=[0, 0, 0],
            ends=[INT_MAX, -1, INT_MAX])
        slice_pos = paddle.slice(
            pos_seq_pad,
            axes=[0, 1, 2],
            starts=[0, 1, 0],
            ends=[INT_MAX, INT_MAX, INT_MAX])
        slice_neg = paddle.slice(
            neg_seq_pad,
            axes=[0, 1, 2],
            starts=[0, 1, 0],
            ends=[INT_MAX, INT_MAX, INT_MAX])
        test_pos = paddle.sum(paddle.sum(paddle.log(self.epsilon + self.sigm(
            paddle.clip(
                paddle.sum(paddle.multiply(slice_gru, slice_pos),
                           axis=2,
                           keepdim=True)))),
                                         axis=2),
                              axis=1,
                              keepdim=True)

        test_neg = paddle.sum(paddle.sum(paddle.log(self.epsilon + self.sigm(
            paddle.clip(
                paddle.sum(paddle.multiply(slice_gru, slice_neg),
                           axis=2,
                           keepdim=True),
                min=-15,
                max=15))),
                                         axis=2),
                              axis=1,
                              keepdim=True)

        aux_loss = paddle.mean(paddle.add(test_neg, test_pos))
        # ------------------------- RNN-gru --------------------------
        prev = paddle.zeros(
            shape=[
                concat_weighted_vector[0].shape[0], self.item_emb_size * 2
            ],
            dtype='float32')
        attention_rnn_res = paddle.zeros(
            shape=[
                0, concat_weighted_vector.shape[1], self.item_emb_size * 2
            ],
            dtype='float32')
        for i in range(concat_weighted_vector.shape[0]):
            word = concat_weighted_vector[i]
            y_out, hidden = self.gru_cell_attention(word, prev)
            prev = hidden
            attention_rnn_res = paddle.concat(
                [attention_rnn_res, hidden.unsqueeze(axis=[0])], 0)
        attention_rnn_res_T = paddle.transpose(attention_rnn_res,
                                               [1, 0, 2])[:, -1, :]
        # ------------------------- top nn Layer --------------------------
        embedding_concat = paddle.concat(
            [attention_rnn_res_T, target_concat], axis=1)
        for layer in self.top_layer:
            embedding_concat = layer(embedding_concat)
        logit = paddle.add(embedding_concat, self.item_b_attr(target_item))
        return logit, aux_loss


class StaticDIENLayer(nn.Layer):
    def __init__(self, item_emb_size, cat_emb_size, act, is_sparse,
                 use_DataLoader, item_count, cat_count):
        super(StaticDIENLayer, self).__init__()

        self.epsilon = 0.00000001
        self.item_emb_size = item_emb_size
        self.cat_emb_size = cat_emb_size
        print()
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
            padding_idx=0,
            name="item_emb")
        self.hist_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="cat_emb")
        self.target_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="item_emb")
        self.target_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="cat_emb")
        self.target_item_seq_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="item_emb")

        self.target_cat_seq_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="cat_emb")

        self.neg_item_seq_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="item_emb")

        self.neg_cat_seq_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            padding_idx=0,
            name="cat_emb")

        self.item_b_attr = paddle.nn.Embedding(
            self.item_count,
            1,
            sparse=self.is_sparse,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)))

        # ------------------------- attention net --------------------------
        self.attention_layer = []
        sizes = [(self.item_emb_size + self.cat_emb_size) * 4
                 ] + [80] + [40] + [1]
        acts = ["sigmoid" for _ in range(len(sizes) - 2)] + [None]

        for i in range(len(sizes) - 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.framework.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()))
            self.add_sublayer('linear_%d' % i, linear)
            self.attention_layer.append(linear)
            if acts[i] == 'sigmoid':
                act = paddle.nn.Sigmoid()
                self.add_sublayer('act_%d' % i, act)
                self.attention_layer.append(act)

        # ------------------------- prev net --------------------------
        self.top_layer = []
        sizes = [(self.item_emb_size + self.cat_emb_size) * 2
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
            self.top_layer.append(linear)
            if acts[i] == 'sigmoid':
                act = paddle.nn.Sigmoid()
                self.add_sublayer('act_%d' % i, act)
                self.top_layer.append(act)

        # ------------------------- gru-net --------------------------
        self.gru_net = paddle.nn.GRU(
            input_size=self.item_emb_size + self.cat_emb_size,
            hidden_size=self.item_emb_size + self.cat_emb_size,
            num_layers=2)
        # paddle暂不支持AUGRU，这里用的普通GRU，不同于论文部分 
        self.gru_cell_attention = paddle.nn.GRUCell(
            self.item_emb_size + self.cat_emb_size,
            self.item_emb_size + self.cat_emb_size)
        self.sigm = paddle.nn.Sigmoid()

        #------------------------- attention net --------------------------

    def forward(self, hist_item_seq, hist_cat_seq, target_item, target_cat,
                label, mask, target_item_seq, target_cat_seq,
                neg_hist_item_seq, neg_hist_cat_seq):
        # ------------------------- network data --------------------------
        # print("---neg_hist_cat_seq----",neg_hist_cat_seq)
        hist_item_emb = self.hist_item_emb_attr(hist_item_seq)
        hist_cat_emb = self.hist_cat_emb_attr(hist_cat_seq)
        target_item_emb = self.target_item_emb_attr(target_item)
        target_cat_emb = self.target_cat_emb_attr(target_cat)
        target_item_seq_emb = self.target_item_seq_emb_attr(target_item_seq)
        target_cat_seq_emb = self.target_cat_seq_emb_attr(target_cat_seq)
        neg_hist_item_emb = self.neg_item_seq_emb_attr(neg_hist_item_seq)
        neg_hist_cat_emb = self.neg_cat_seq_emb_attr(neg_hist_cat_seq)
        item_b = self.item_b_attr(target_item)

        # ------------------------- Interest Extractor Layer --------------------------
        hist_seq_concat = paddle.concat([hist_item_emb, hist_cat_emb], axis=2)
        neg_hist_seq_concat = paddle.concat(
            [neg_hist_item_emb, neg_hist_cat_emb], axis=2)
        target_seq_concat = paddle.concat(
            [target_item_seq_emb, target_cat_seq_emb], axis=2)
        target_concat = paddle.concat(
            [target_item_emb, target_cat_emb], axis=1)
        gru_shape = hist_seq_concat.shape
        reshape_hist_item_emb = hist_seq_concat

        neg_reshape_hist_item_emb = neg_hist_seq_concat
        gru_hist_item_emb = hist_seq_concat
        gru_out, gru_hid = self.gru_net(gru_hist_item_emb)
        # ------------------------- attention --------------------------

        concat = paddle.concat(
            [
                hist_seq_concat, target_seq_concat,
                paddle.subtract(hist_seq_concat, target_seq_concat),
                paddle.multiply(hist_seq_concat, target_seq_concat)
            ],
            axis=2)

        for attlayer in self.attention_layer:
            concat = attlayer(concat)

        atten_fc3 = paddle.add(concat, mask)  #concat + mask  #concat + mask
        atten_fc3 = paddle.transpose(atten_fc3, perm=[0, 2, 1])
        atten_fc3 = paddle.scale(
            atten_fc3, scale=(self.item_emb_size + self.cat_emb_size)**-0.5)
        weight = paddle.nn.functional.softmax(atten_fc3)
        weighted = paddle.transpose(x=weight, perm=[0, 2, 1])
        weighted_vector = paddle.multiply(weighted, hist_seq_concat)
        weighted_vector = paddle.transpose(weighted_vector, perm=[1, 0, 2])
        # ------------------------- rnn-gru --------------------------
        concat_weighted_vector = paddle.concat([weighted_vector], axis=2)
        # ------------------------- Auxiliary loss  --------------------------
        start_value = paddle.zeros(shape=[1], dtype="float32")
        gru_out_pad = gru_out

        pos_seq_pad = reshape_hist_item_emb
        neg_seq_pad = neg_reshape_hist_item_emb

        INT_MAX = int(1.0 * 1e9)
        slice_gru = paddle.slice(
            gru_out_pad,
            axes=[0, 1, 2],
            starts=[0, 0, 0],
            ends=[INT_MAX, -1, INT_MAX])
        slice_pos = paddle.slice(
            pos_seq_pad,
            axes=[0, 1, 2],
            starts=[0, 1, 0],
            ends=[INT_MAX, INT_MAX, INT_MAX])
        slice_neg = paddle.slice(
            neg_seq_pad,
            axes=[0, 1, 2],
            starts=[0, 1, 0],
            ends=[INT_MAX, INT_MAX, INT_MAX])

        test_pos = paddle.sum(paddle.sum(paddle.log(self.epsilon + self.sigm(
            paddle.clip(
                paddle.sum(paddle.multiply(slice_gru, slice_pos),
                           axis=2,
                           keepdim=True)))),
                                         axis=2),
                              axis=1,
                              keepdim=True)

        test_neg = paddle.sum(paddle.sum(paddle.log(self.epsilon + self.sigm(
            paddle.clip(
                paddle.sum(paddle.multiply(slice_gru, slice_neg),
                           axis=2,
                           keepdim=True),
                min=-15,
                max=15))),
                                         axis=2),
                              axis=1,
                              keepdim=True)
        aux_loss = paddle.mean(paddle.add(test_neg, test_pos))

        # ------------------------- RNN-gru --------------------------
        self.rnn = paddle.nn.RNN(cell=self.gru_cell_attention, time_major=True)
        attention_rnn_res, final_states= self.rnn(inputs=concat_weighted_vector)
        attention_rnn_res_T = paddle.transpose(attention_rnn_res,
                                               [1, 0, 2])[:, -1, :]

        # ------------------------- top nn Layer --------------------------
        embedding_concat = paddle.concat(
            [attention_rnn_res_T, target_concat], axis=1)
        for layer in self.top_layer:
            embedding_concat = layer(embedding_concat)
        logit = paddle.add(embedding_concat, self.item_b_attr(target_item))

        return logit, aux_loss
