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
import paddle.fluid as fluid
import numpy as np
import paddle.nn.functional as F

class StaticDIENLayer(nn.Layer):
    def __init__(self, item_emb_size, cat_emb_size, act, is_sparse,
                 use_DataLoader, item_count, cat_count):
        super(StaticDIENLayer, self).__init__()

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

        self.neg_item_seq_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=self.is_sparse,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()),
            name="item_emb")

        self.neg_cat_seq_emb_attr = paddle.nn.Embedding(
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

#  # ------------------------- prev net --------------------------
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

#  # ------------------------- gru-net --------------------------

        self.rnn = fluid.layers.StaticRNN(name="attention_evolution")
        self.gru_net = paddle.nn.GRU(input_size=self.item_emb_size + self.cat_emb_size, hidden_size=self.item_emb_size + self.cat_emb_size, num_layers=2)
        self.gru_cell_attention = paddle.nn.GRUCell(self.item_emb_size + self.cat_emb_size, self.item_emb_size + self.cat_emb_size)
        self.sigm = paddle.nn.Sigmoid()
#  # ------------------------- attention net --------------------------
#         self.con_layer = []

#         self.firInDim = self.item_emb_size + self.cat_emb_size
#         self.firOutDim = self.item_emb_size + self.cat_emb_size

#         linearCon = paddle.nn.Linear(
#             in_features=self.firInDim,
#             out_features=self.firOutDim,
#             weight_attr=paddle.framework.ParamAttr(
#                 initializer=paddle.nn.initializer.XavierUniform()),
#             bias_attr=paddle.ParamAttr(
#                 initializer=paddle.nn.initializer.Constant(value=0.0)))
#         self.add_sublayer('linearCon', linearCon)
#         self.con_layer.append(linearCon)

#         conDim = self.item_emb_size + self.cat_emb_size + self.item_emb_size + self.cat_emb_size

#         conSizes = [conDim] + [80] + [40] + [1]
#         conActs = ["sigmoid" for _ in range(len(conSizes) - 2)] + [None]

#         for i in range(len(conSizes) - 1):
#             linear = paddle.nn.Linear(
#                 in_features=conSizes[i],
#                 out_features=conSizes[i + 1],
#                 weight_attr=paddle.framework.ParamAttr(
#                     initializer=paddle.nn.initializer.XavierUniform()),
#                 bias_attr=paddle.ParamAttr(
#                     initializer=paddle.nn.initializer.Constant(value=0.0)))
#             self.add_sublayer('linear_%d' % i, linear)
#             self.con_layer.append(linear)
#             if conActs[i] == 'sigmoid':
#                 act = paddle.nn.Sigmoid()
#                 self.add_sublayer('act_%d' % i, act)
#                 self.con_layer.append(act)


    def forward(self, hist_item_seq, hist_cat_seq, target_item, target_cat,
                label, mask, target_item_seq, target_cat_seq,neg_hist_item_seq,neg_hist_cat_seq):
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
        # batch_size = self.train_batch_size

        # ------------------------- Interest Extractor Layer --------------------------
        hist_seq_concat = paddle.concat([hist_item_emb, hist_cat_emb], axis=2)
        neg_hist_seq_concat = paddle.concat([neg_hist_item_emb, neg_hist_cat_emb], axis=2)
        target_seq_concat = paddle.concat(
            [target_item_seq_emb, target_cat_seq_emb], axis=2)
        target_concat = paddle.concat(
            [target_item_emb, target_cat_emb], axis=1)
        # fluid.layers.Print(hist_seq_concat, message="----hist_seq_concat----")
        gru_shape = hist_seq_concat.shape
        # print("------gru_shape-----", gru_shape)
        # fluid.layers.Print(gru_shape, message="----gru_shape---")
        # reshape_hist_item_emb = paddle.sum(hist_seq_concat, axis=1)
        reshape_hist_item_emb = hist_seq_concat
        # fluid.layers.Print(reshape_hist_item_emb, message="----reshape_hist_item_emb--sum1--")
        # neg_reshape_hist_item_emb = paddle.sum(neg_hist_seq_concat,axis=1)
        neg_reshape_hist_item_emb = neg_hist_seq_concat
        gru_hist_item_emb=hist_seq_concat
        # gru_in_shape = gru_hist_item_emb.shape
        # fluid.layers.Print(gru_hist_item_emb, message="--gru_hist_item_emb---")
        # gru_hist_item_emb = paddle.concat(hist_seq_concat,axis=1)
        # self.gru_net = paddle.nn.GRU(gru_in_shape[0], gru_in_shape[1], self.item_emb_size * 2)
        gru_out, gru_hid = self.gru_net(gru_hist_item_emb)
        # print("----gru_out----", gru_out)
        # print("neg_hist_seq_concat",neg_hist_seq_concat)
        # print("neg_reshape_hist_item_emb",neg_reshape_hist_item_emb)
        # fluid.layers.Print(neg_reshape_hist_item_emb, message="[32, 128]-sum---neg_reshape_hist_item_emb")
        # fluid.layers.Print(neg_hist_seq_concat, message="[32, 32, 128]----neg_hist_seq_concat")
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

        # fluid.layers.Print(concat, message="----concat---")
        # fluid.layers.Print(mask, message="----mask---")
        atten_fc3 = paddle.add(concat, mask) #concat + mask  #concat + mask
        atten_fc3 = paddle.transpose(atten_fc3, perm=[0, 2, 1])
        atten_fc3 = paddle.scale(atten_fc3, scale=(self.item_emb_size + self.cat_emb_size)**-0.5)
        weight = paddle.nn.functional.softmax(atten_fc3)
        weighted = paddle.transpose(x=weight, perm=[0,2,1])
        weighted_vector = paddle.multiply(weighted, hist_seq_concat)
        # weighted_vector = paddle.matmul(weighted, hist_seq_concat)
        # output = paddle.matmul(weight, hist_seq_concat)
        # output = paddle.reshape(output, shape=[0, self.firInDim])
        weighted_vector = paddle.transpose(weighted_vector, perm=[1, 0, 2])
        # ------------------------- rnn-gru --------------------------
        concat_weighted_vector = paddle.concat(
            [weighted_vector], axis=2)
        # ------------------------- Auxiliary loss  --------------------------
        start_value = paddle.zeros(shape=[1],dtype="float32")
        # pad_value = paddle.zeros(shape=[1],dtype="float32")        
        # gru_out_pad = F.pad(gru_out, value=0,  mode='constant')
        # pos_seq_pad = F.pad(reshape_hist_item_emb, value=0,  mode='constant')
        # neg_seq_pad = F.pad(neg_reshape_hist_item_emb, value=0,  mode='constant')
        # seq_shape = paddle.shape(pos_seq_pad)
        # pad_value = fluid.layers.zeros(shape=[1], dtype='float32')
        # start_value = fluid.layers.zeros(shape=[1], dtype='int32')
        # print("pad_value",pad_value)
        # seq_shape = gru_out.shape
        # print("---seq_shape--",seq_shape)
        # fluid.layers.Print(seq_shape, message="---seq_shape---")
        # fluid.layers.Print(seq_shape[1], message="---seq_shape[1]---")

        # gru_out_pad = paddle.reshape(gru_out, [seq_shape[0]*seq_shape[1],seq_shape[2]]) # ???????
        # print(gru_out)
        # print(self.item_emb_size * 2)
        # gru_out_pad = paddle.reshape(gru_out,[-1,self.item_emb_size * 2])
        gru_out_pad = gru_out
        # gru_out_pad, lengths = fluid.layers.sequence_pad(gru_out, pad_value)
        # print("gru_out_pad",gru_out_pad)

        # fluid.layers.Print(gru_out_pad, message="gru_out_pad")
        # fluid.layers.Print(lengths, message="lengths")

        pos_seq_pad = reshape_hist_item_emb
        neg_seq_pad = neg_reshape_hist_item_emb
        # pos_seq_pad, _ = fluid.layers.sequence_pad(reshape_hist_item_emb,
        #                                            pad_value)
        # neg_seq_pad, _ = fluid.layers.sequence_pad(neg_reshape_hist_item_emb,
        #                                            pad_value)
        # fluid.layers.Print(gru_out, message="---gru_out----")
        # fluid.layers.Print(gru_out_pad, message = "----gru_out_pad---")  #[batch_size,time_steps,num_directions * hidden_size]
        # fluid.layers.Print(neg_seq_pad, message = "----neg_seq_pad---")
        # print("gru_out_pad", gru_out_pad)
        # print("neg_seq_pad",neg_seq_pad)
        # a=gru_out_pad[:, start_value:seq_shape[1] - 1, :]
        # b=neg_seq_pad[:, start_value + 1:seq_shape[1], :]
        # fluid.layers.Print(a, message = "----gru_out_pad[:, start_value:seq_shape[1] - 1, :]---")
        # fluid.layers.Print(b, message = "----neg_seq_pad[:, start_value + 1:seq_shape[1], :]---")

        # gru_index = fluid.layers.data(name="gru_index", shape=[-1], dtype='int64')

        # gru_index = fluid.layers.embedding(
        #     input=x,
        #     size=[vocab_size, hidden_size],
        #     dtype='float32',
        #     is_sparse=False)

        # gru_out_pad = paddle.index_sample(gru_out, gru_index). ?????????
        # seq_len = gru_shape[1]
        # gru_out_pad = paddle.index_sample(gru_out_pad, [:,seq_len,:])
        # gru_out_pad = gru_out_pad[]
        # print(gru_shape[1])
        # print(gru_out_pad[:, start_value:gru_shape[1] - 1, :])

        INT_MAX = int(1.0*1e9)
        slice_gru = paddle.slice(gru_out_pad,axes=[0,1,2],starts=[0,0,0],ends=[INT_MAX,-1,INT_MAX])
        slice_pos = paddle.slice(pos_seq_pad,axes=[0,1,2],starts=[0,1,0],ends=[INT_MAX,INT_MAX,INT_MAX])
        slice_neg = paddle.slice(neg_seq_pad,axes=[0,1,2],starts=[0,1,0],ends=[INT_MAX,INT_MAX,INT_MAX])
        # print("----slice_neg----",slice_neg)
        # paddle.static.Print(slice_neg, message = "----slice_neg----")
        # print("----slice_pos----",slice_pos)
        # paddle.static.Print(slice_pos, message = "----slice_pos----")
        # print("----slice_gru----",slice_gru)
        # paddle.static.Print(slice_gru, message = "----slice_gru----")
        # slice_gru = paddle.slice(gru_out_pad, [:,-1,:])
        # slice_pos = paddle.slice(gru_out_pad, [:,1:,:])
        test_pos = paddle.sum(
            paddle.sum(
                paddle.log(
                    self.sigm(
                        paddle.sum(
                            paddle.multiply(slice_gru,slice_pos),
                            # paddle.multiply(gru_out_pad[:, start_value:seq_shape[1] - 1, :], pos_seq_pad[:, start_value + 1:seq_shape[1], :]),
                            axis=2,
                            keepdim=True))),
                axis=2),
            axis=1,
            keepdim=True)
        
        test_neg = paddle.sum(
            paddle.sum(
                paddle.log(
                    self.sigm(
                        paddle.sum(
                            paddle.multiply(slice_gru, slice_neg),
                            # paddle.multiply(gru_out_pad[:, start_value:seq_shape[1] - 1, :], neg_seq_pad[:, start_value + 1:seq_shape[1], :]),
                            axis=2,
                            keepdim=True))),
                axis=2),
            axis=1,
            keepdim=True)
        aux_loss=paddle.mean(paddle.add(test_neg, test_pos))


        # ------------------------- RNN-gru --------------------------

        with self.rnn.step():
            word = self.rnn.step_input(concat_weighted_vector)
            prev = self.rnn.memory(
                shape=[-1, self.item_emb_size * 2], batch_ref=word)
            # hidden, _, _ = fluid.layers.gru_unit(
            #     input=word, hidden=prev, size=self.item_emb_size * 6)
            y_out, hidden = self.gru_cell_attention(word, prev)
            # hidden, _, _ = paddle.nn.GRUCell(
            #     input=word, hidden=prev, size=self.item_emb_size * 6)
            self.rnn.update_memory(prev, hidden)
            self.rnn.output(hidden)
        attention_rnn_res = self.rnn()
        attention_rnn_res_T = fluid.layers.transpose(attention_rnn_res,
                                                     [1, 0, 2])[:, -1, :]
        # ------------------------- top nn Layer --------------------------
        embedding_concat = fluid.layers.concat(
            [attention_rnn_res_T, target_concat], axis=1)
        # paddle.static.Print(embedding_concat)
        for layer in self.top_layer:
            embedding_concat = layer(embedding_concat)
        logit = paddle.add(embedding_concat,self.item_b_attr(target_item))

        return logit, aux_loss
