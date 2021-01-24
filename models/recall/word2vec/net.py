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


class Word2VecLayer(nn.Layer):
    def __init__(self, sparse_feature_number, emb_dim, neg_num, emb_name,
                 emb_w_name, emb_b_name):
        super(Word2VecLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.emb_dim = emb_dim
        self.neg_num = neg_num
        self.emb_name = emb_name
        self.emb_w_name = emb_w_name
        self.emb_b_name = emb_b_name

        init_width = 0.5 / self.emb_dim
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.emb_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name=self.emb_name,
                initializer=paddle.nn.initializer.Uniform(-init_width,
                                                          init_width)))

        self.embedding_w = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.emb_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name=self.emb_w_name,
                initializer=paddle.nn.initializer.Constant(value=0.0)))

        self.embedding_b = paddle.nn.Embedding(
            self.sparse_feature_number,
            1,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name=self.emb_b_name,
                initializer=paddle.nn.initializer.Constant(value=0.0)))

    def forward(self, inputs):
        input_emb = self.embedding(inputs[0])
        true_emb_w = self.embedding_w(inputs[1])
        true_emb_b = self.embedding_b(inputs[1])
        input_emb = paddle.squeeze(x=input_emb, axis=[1])
        true_emb_w = paddle.squeeze(x=true_emb_w, axis=[1])
        true_emb_b = paddle.squeeze(x=true_emb_b, axis=[1])

        neg_emb_w = self.embedding_w(inputs[2])
        neg_emb_b = self.embedding_b(inputs[2])

        neg_emb_b_vec = paddle.reshape(neg_emb_b, shape=[-1, self.neg_num])

        true_logits = paddle.add(x=paddle.sum(x=paddle.multiply(
            x=input_emb, y=true_emb_w),
                                              axis=1,
                                              keepdim=True),
                                 y=true_emb_b)

        input_emb_re = paddle.reshape(input_emb, shape=[-1, 1, self.emb_dim])
        neg_matmul = paddle.matmul(input_emb_re, neg_emb_w, transpose_y=True)
        neg_matmul_re = paddle.reshape(neg_matmul, shape=[-1, self.neg_num])
        neg_logits = paddle.add(x=neg_matmul_re, y=neg_emb_b_vec)

        return true_logits, neg_logits


class Word2VecInferLayer(nn.Layer):
    def __init__(self, sparse_feature_number, emb_dim, emb_name):
        super(Word2VecInferLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.emb_dim = emb_dim
        self.emb_name = emb_name

        init_width = 0.5 / self.emb_dim
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.emb_dim,
            weight_attr=paddle.ParamAttr(
                name=self.emb_name,
                initializer=paddle.nn.initializer.Uniform(-init_width,
                                                          init_width)))

    def forward(self, analogy_a, analogy_b, analogy_c, all_label):
        emb_a = self.embedding(analogy_a)
        emb_b = self.embedding(analogy_b)
        emb_c = self.embedding(analogy_c)
        emb_all_label = self.embedding(all_label)

        target = emb_b - emb_a + emb_c
        emb_all_label_l2 = F.normalize(emb_all_label, axis=1)
        dist = paddle.matmul(x=target, y=emb_all_label_l2, transpose_y=True)
        values, pred_idx = paddle.topk(x=dist, k=4)
        return values, pred_idx
