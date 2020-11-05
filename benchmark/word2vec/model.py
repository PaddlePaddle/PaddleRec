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

import math
import paddle
import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.dense_feature_dim = envs.get_global_env(
            "hyper_parameters.dense_feature_dim")
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim")
        self.learning_rate = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate")

    def input_data(self, is_infer=False, **kwargs):

        input_word = fluid.layers.data(
            name="input_word", shape=[1], dtype='int64', lod_level=1)
        true_word = fluid.layers.data(
            name='true_label', shape=[1], dtype='int64', lod_level=1)
        neg_word = fluid.layers.data(
            name="neg_label", shape=[1], dtype='int64', lod_level=1)
        inputs = [input_word, true_word, neg_word]
        return inputs

    def net(self, input, is_infer=False):

        init_width = 0.5 / params.embedding_size
        input_emb = fluid.layers.embedding(
            input=inputs[0],
            is_sparse=params.is_sparse,
            size=[params.dict_size, params.embedding_size],
            param_attr=fluid.ParamAttr(
                name='emb',
                initializer=fluid.initializer.Uniform(-init_width,
                                                      init_width)))

        true_emb_w = fluid.layers.embedding(
            input=inputs[1],
            is_sparse=params.is_sparse,
            size=[params.dict_size, params.embedding_size],
            param_attr=fluid.ParamAttr(
                name='emb_w',
                initializer=fluid.initializer.Constant(value=0.0)))

        true_emb_b = fluid.layers.embedding(
            input=inputs[1],
            is_sparse=params.is_sparse,
            size=[params.dict_size, 1],
            param_attr=fluid.ParamAttr(
                name='emb_b',
                initializer=fluid.initializer.Constant(value=0.0)))

        neg_word_reshape = fluid.layers.reshape(inputs[2], shape=[-1, 1])
        neg_word_reshape.stop_gradient = True

        neg_emb_w = fluid.layers.embedding(
            input=neg_word_reshape,
            is_sparse=params.is_sparse,
            size=[params.dict_size, params.embedding_size],
            param_attr=fluid.ParamAttr(
                name='emb_w', learning_rate=1.0))

        neg_emb_w_re = fluid.layers.reshape(
            neg_emb_w, shape=[-1, params.nce_num, params.embedding_size])

        neg_emb_b = fluid.layers.embedding(
            input=neg_word_reshape,
            is_sparse=params.is_sparse,
            size=[params.dict_size, 1],
            param_attr=fluid.ParamAttr(
                name='emb_b', learning_rate=1.0))

        neg_emb_b_vec = fluid.layers.reshape(
            neg_emb_b, shape=[-1, params.nce_num])

        true_logits = fluid.layers.elementwise_add(
            fluid.layers.reduce_sum(
                fluid.layers.elementwise_mul(input_emb, true_emb_w),
                dim=1,
                keep_dim=True),
            true_emb_b)

        input_emb_re = fluid.layers.reshape(
            input_emb, shape=[-1, 1, params.embedding_size])

        neg_matmul = fluid.layers.matmul(
            input_emb_re, neg_emb_w_re, transpose_y=True)
        neg_matmul_re = fluid.layers.reshape(
            neg_matmul, shape=[-1, params.nce_num])
        neg_logits = fluid.layers.elementwise_add(neg_matmul_re, neg_emb_b_vec)
        # nce loss

        label_ones = fluid.layers.fill_constant_batch_size_like(
            true_logits, shape=[-1, 1], value=1.0, dtype='float32')
        label_zeros = fluid.layers.fill_constant_batch_size_like(
            true_logits,
            shape=[-1, params.nce_num],
            value=0.0,
            dtype='float32')

        true_xent = fluid.layers.sigmoid_cross_entropy_with_logits(true_logits,
                                                                   label_ones)
        neg_xent = fluid.layers.sigmoid_cross_entropy_with_logits(neg_logits,
                                                                  label_zeros)
        cost = fluid.layers.elementwise_add(
            fluid.layers.reduce_sum(
                true_xent, dim=1),
            fluid.layers.reduce_sum(
                neg_xent, dim=1))
        avg_cost = fluid.layers.reduce_mean(cost)

    def optimizer(self):
        optimizer = paddle.optimizer.Adam(self.learning_rate, lazy_mode=True)
        return optimizer

    def infer_net(self):
        pass
