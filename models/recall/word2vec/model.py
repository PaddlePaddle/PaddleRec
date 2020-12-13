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

import numpy as np
import paddle

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase
from word2vec_net import Word2VecLayer


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.is_distributed = True if envs.get_fleet_mode().upper(
        ) == "PSLIB" else False
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim")
        self.neg_num = envs.get_global_env("hyper_parameters.neg_num")
        self.with_shuffle_batch = envs.get_global_env(
            "hyper_parameters.with_shuffle_batch")
        self.learning_rate = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate")
        self.decay_steps = envs.get_global_env(
            "hyper_parameters.optimizer.decay_steps")
        self.decay_rate = envs.get_global_env(
            "hyper_parameters.optimizer.decay_rate")

    def input_data(self, is_infer=False, **kwargs):
        if is_infer:
            analogy_a = paddle.static.data(
                name="analogy_a", shape=[None], dtype='int64')
            analogy_b = paddle.static.data(
                name="analogy_b", shape=[None], dtype='int64')
            analogy_c = paddle.static.data(
                name="analogy_c", shape=[None], dtype='int64')
            analogy_d = paddle.static.data(
                name="analogy_d", shape=[None], dtype='int64')
            return [analogy_a, analogy_b, analogy_c, analogy_d]

        input_word = paddle.static.data(
            name="input_word", shape=[None, 1], dtype='int64')
        true_word = paddle.static.data(
            name='true_label', shape=[None, 1], dtype='int64')
        if self.with_shuffle_batch:
            return [input_word, true_word]

        neg_word = paddle.static.data(
            name="neg_label", shape=[None, self.neg_num], dtype='int64')
        return [input_word, true_word, neg_word]

    def net(self, inputs, is_infer=False):
        if is_infer:
            self.infer_net(inputs)
            return

        word2vec_model = Word2VecLayer(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            self.neg_num,
            emb_name="emb",
            emb_w_name="emb_w",
            emb_b_name="emb_b")
        true_logits, neg_logits = word2vec_model.forward(inputs)

        label_ones = paddle.full(
            shape=[paddle.shape(true_logits)[0], 1], fill_value=1.0)
        label_zeros = paddle.full(
            shape=[paddle.shape(true_logits)[0], self.neg_num], fill_value=0.0)

        true_xent = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(
            true_logits, label_ones)
        neg_xent = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(
            neg_logits, label_zeros)
        cost = paddle.add(x=paddle.sum(x=true_xent, axis=1),
                          y=paddle.sum(x=neg_xent, axis=1))
        avg_cost = paddle.mean(x=cost)

        self._cost = avg_cost
        #global_right_cnt = paddle.fluid.layers.create_global_var(
        #    name="global_right_cnt",
        #    persistable=True,
        #    dtype='float32',
        #    shape=[1],
        #    value=0)
        #global_total_cnt = paddle.fluid.layers.create_global_var(
        #    name="global_total_cnt",
        #    persistable=True,
        #    dtype='float32',
        #    shape=[1],
        #    value=0)
        #global_right_cnt.stop_gradient = True
        #global_total_cnt.stop_gradient = True
        self._metrics["LOSS"] = avg_cost

    def optimizer(self):
        optimizer = paddle.fluid.optimizer.SGD(
            learning_rate=paddle.fluid.layers.exponential_decay(
                learning_rate=self.learning_rate,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate,
                staircase=True))
        return optimizer

    def infer_net(self, inputs):
        def embedding_layer(input, table_name, initializer_instance=None):
            emb = paddle.static.nn.embedding(
                input=input,
                size=[self.sparse_feature_number, self.sparse_feature_dim],
                param_attr=table_name)
            return emb

        all_label = np.arange(self.sparse_feature_number).reshape(
            self.sparse_feature_number).astype('int32')
        self.all_label = paddle.cast(
            x=paddle.nn.functional.assign(all_label), dtype='int64')
        emb_all_label = embedding_layer(self.all_label, "emb")
        emb_a = embedding_layer(inputs[0], "emb")
        emb_b = embedding_layer(inputs[1], "emb")
        emb_c = embedding_layer(inputs[2], "emb")

        target = paddle.add(
            x=paddle.fluid.layers.nn.elementwise_sub(emb_b, emb_a), y=emb_c)

        emb_all_label_l2 = paddle.fluid.layers.l2_normalize(
            x=emb_all_label, axis=1)
        dist = paddle.fluid.layers.matmul(
            x=target, y=emb_all_label_l2, transpose_y=True)
        values, pred_idx = paddle.topk(x=dist, k=1)
        label = paddle.fluid.layers.expand(
            paddle.unsqueeze(
                inputs[3], axis=[1]), expand_times=[1, 1])
        label_ones = paddle.fluid.layers.fill_constant_batch_size_like(
            label, shape=[-1, 1], value=1.0, dtype='float32')
        right_cnt = paddle.sum(x=paddle.cast(
            paddle.equal(
                x=pred_idx, y=label), dtype='float32'))
        total_cnt = paddle.sum(x=label_ones)

        global_right_cnt = paddle.fluid.layers.create_global_var(
            name="global_right_cnt",
            persistable=True,
            dtype='float32',
            shape=[1],
            value=0)
        global_total_cnt = paddle.fluid.layers.create_global_var(
            name="global_total_cnt",
            persistable=True,
            dtype='float32',
            shape=[1],
            value=0)
        global_right_cnt.stop_gradient = True
        global_total_cnt.stop_gradient = True

        tmp1 = paddle.add(x=right_cnt, y=global_right_cnt)
        paddle.nn.functional.assign(tmp1, global_right_cnt)
        tmp2 = paddle.add(x=total_cnt, y=global_total_cnt)
        paddle.nn.functional.assign(tmp2, global_total_cnt)

        acc = paddle.divide(
            x=global_right_cnt, y=global_total_cnt, name="total_acc")
        self._infer_results['acc'] = acc
