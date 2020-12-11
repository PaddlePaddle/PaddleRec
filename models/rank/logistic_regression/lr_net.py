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


class LRLayer(nn.Layer):
    def __init__(self, sparse_feature_number, init_value, reg, num_field):
        super(LRLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.init_value_ = init_value
        self.reg = reg
        self.num_field = num_field
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number + 1,
            1,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0, std=self.init_value_),
                regularizer=paddle.regularizer.L1Decay(self.reg)))

        self.b_linear = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0))

    def forward(self, input_idx, input_weights):
        emb = self.embedding(input_idx)
        emb_re = paddle.reshape(
            emb, shape=[-1, self.num_field])  # None * num_field * 1
        y_first_order = paddle.sum(emb_re * input_weights, 1, keepdim=True)

        predict = paddle.nn.functional.sigmoid(y_first_order + self.b_linear)
        return predict
