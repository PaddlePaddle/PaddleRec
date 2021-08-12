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


class DSSMLayer(nn.Layer):
    def __init__(self, trigram_d, neg_num, slice_end, hidden_layers,
                 hidden_acts):
        super(DSSMLayer, self).__init__()

        self.hidden_layers = [trigram_d] + hidden_layers
        self.hidden_acts = hidden_acts
        self.slice_end = slice_end

        self._query_layers = []
        for i in range(len(self.hidden_layers) - 1):
            linear = paddle.nn.Linear(
                in_features=self.hidden_layers[i],
                out_features=self.hidden_layers[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierNormal(
                        fan_in=self.hidden_layers[i],
                        fan_out=self.hidden_layers[i + 1])),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierNormal(
                        fan_in=self.hidden_layers[i],
                        fan_out=self.hidden_layers[i + 1])))
            self.add_sublayer('query_linear_%d' % i, linear)
            self._query_layers.append(linear)
            if self.hidden_acts[i] == "relu":
                act = paddle.nn.ReLU()
                self.add_sublayer('query_act_%d' % i, act)
                self._query_layers.append(act)

        self._doc_layers = []
        for i in range(len(self.hidden_layers) - 1):
            linear = paddle.nn.Linear(
                in_features=self.hidden_layers[i],
                out_features=self.hidden_layers[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierNormal(
                        fan_in=self.hidden_layers[i],
                        fan_out=self.hidden_layers[i + 1])),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierNormal(
                        fan_in=self.hidden_layers[i],
                        fan_out=self.hidden_layers[i + 1])))
            self.add_sublayer('pos_linear_%d' % i, linear)
            self._doc_layers.append(linear)
            if self.hidden_acts[i] == "relu":
                act = paddle.nn.ReLU()
                self.add_sublayer('pos_act_%d' % i, act)
                self._doc_layers.append(act)

    def forward(self, input_data, is_infer):
        query_fc = input_data[0]
        for n_layer in self._query_layers:
            query_fc = n_layer(query_fc)
        self.query_fc = query_fc

        doc_pos_fc = input_data[1]
        for n_layer in self._doc_layers:
            doc_pos_fc = n_layer(doc_pos_fc)
        self.doc_pos_fc = doc_pos_fc

        self.params = [self._query_layers[-2].bias]

        R_Q_D_p = F.cosine_similarity(
            query_fc, doc_pos_fc, axis=1).reshape([-1, 1])

        if is_infer:
            return R_Q_D_p, paddle.ones(shape=[self.slice_end, 1])

        R_Q_D_ns = []
        for i in range(len(input_data) - 2):
            doc_neg_fc_i = input_data[i + 2]
            for n_layer in self._doc_layers:
                doc_neg_fc_i = n_layer(doc_neg_fc_i)
            R_Q_D_n = F.cosine_similarity(
                query_fc, doc_neg_fc_i, axis=1).reshape([-1, 1])
            R_Q_D_ns.append(R_Q_D_n)
        concat_Rs = paddle.concat(x=[R_Q_D_p] + R_Q_D_ns, axis=1)
        prob = F.softmax(concat_Rs, axis=1)
        hit_prob = paddle.slice(
            prob, axes=[0, 1], starts=[0, 0], ends=[self.slice_end, 1])
        return R_Q_D_p, hit_prob
