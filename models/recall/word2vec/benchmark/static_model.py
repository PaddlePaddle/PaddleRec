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
import paddle.distributed.fleet as fleet
import math
import numpy as np


class StaticModel(object):
    def __init__(self, config):
        self.cost = None
        self.metrics = {}
        self.config = config
        self.init_hyper_parameters()
        self.optimizer = None

    def init_hyper_parameters(self):
        self.sparse_feature_number = self.config.get(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = self.config.get(
            "hyper_parameters.sparse_feature_dim")
        self.neg_num = self.config.get("hyper_parameters.neg_num")
        self.with_shuffle_batch = self.config.get(
            "hyper_parameters.with_shuffle_batch")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.decay_steps = self.config.get(
            "hyper_parameters.optimizer.decay_steps")
        self.decay_rate = self.config.get(
            "hyper_parameters.optimizer.decay_rate")

    def create_feeds(self, is_infer=False, **kwargs):
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
        init_width = 0.5 / self.sparse_feature_dim

        input_emb = paddle.static.nn.embedding(
            input=inputs[0],
            is_sparse=True,
            size=[self.sparse_feature_number, self.sparse_feature_dim],
            param_attr=paddle.ParamAttr(
                name='emb',
                initializer=paddle.initializer.Uniform(-init_width,
                                                       init_width)))

        true_emb_w = paddle.static.nn.embedding(
            input=inputs[1],
            is_sparse=True,
            size=[self.sparse_feature_number, self.sparse_feature_dim],
            param_attr=paddle.ParamAttr(
                name='emb_w',
                initializer=paddle.initializer.Constant(value=0.0)))

        true_emb_b = paddle.static.nn.embedding(
            input=inputs[1],
            is_sparse=True,
            size=[self.sparse_feature_number, 1],
            param_attr=paddle.ParamAttr(
                name='emb_b',
                initializer=paddle.initializer.Constant(value=0.0)))

        neg_word_reshape = paddle.reshape(inputs[2], shape=[-1, 1])
        neg_word_reshape.stop_gradient = True

        neg_emb_w = paddle.static.nn.embedding(
            input=neg_word_reshape,
            is_sparse=True,
            size=[self.sparse_feature_number, self.sparse_feature_dim],
            param_attr=paddle.ParamAttr(
                name='emb_w', learning_rate=1.0))

        neg_emb_w_re = paddle.reshape(
            neg_emb_w, shape=[-1, self.neg_num, self.sparse_feature_dim])

        neg_emb_b = paddle.static.nn.embedding(
            input=neg_word_reshape,
            is_sparse=True,
            size=[self.sparse_feature_number, 1],
            param_attr=paddle.ParamAttr(
                name='emb_b', learning_rate=1.0))

        neg_emb_b_vec = paddle.reshape(neg_emb_b, shape=[-1, self.neg_num])

        true_logits = paddle.add(paddle.sum(
            paddle.multiply(input_emb, true_emb_w), dim=1, keep_dim=True),
                                 true_emb_b)

        input_emb_re = paddle.reshape(
            input_emb, shape=[-1, 1, self.sparse_feature_dim])

        neg_matmul = paddle.matmul(
            input_emb_re, neg_emb_w_re, transpose_y=True)
        neg_matmul_re = paddle.reshape(neg_matmul, shape=[-1, self.neg_num])
        neg_logits = paddle.add(neg_matmul_re, neg_emb_b_vec)
        # nce loss

        label_ones = paddle.full_like(
            true_logits, fill_value=1.0, dtype='float32')
        label_zeros = paddle.full_like(
            true_logits, fill_value=0.0, dtype='float32')

        true_xent = paddle.nn.functional.binary_cross_entropy(true_logits,
                                                              label_ones)
        neg_xent = paddle.nn.functional.binary_cross_entropy(neg_logits,
                                                             label_zeros)
        cost = paddle.add(paddle.sum(true_xent, dim=1),
                          paddle.sum(neg_xent, dim=1))
        avg_cost = paddle.mean(cost)

        self.inference_target_var = avg_cost
        self.cost = avg_cost
        self.metrics["LOSS"] = avg_cost

        return self.metrics

    def create_optimizer(self, strategy=None):
        pure_bf16 = self.config.get("pure_bf16")
        lr = float(self.config.get("hyper_parameters.optimizer.learning_rate"))
        decay_rate = float(
            self.config.get("hyper_parameters.optimizer.decay_rate"))
        decay_steps = int(
            self.config.get("hyper_parameters.optimizer.decay_steps"))

        # single
        optimizer = paddle.optimizer.SGD(
            learning_rate=paddle.static.exponential_decay(
                learning_rate=lr,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True))

        if strategy != None:
            sync_mode = self.config.get("runner.sync_mode")
            print("sync_mode: {}".format(sync_mode))
            # geo
            if sync_mode == "geo":
                decay_steps = int(decay_steps / fleet.worker_num())
                optimizer = paddle.optimizer.SGD(
                    learning_rate=paddle.static.exponential_decay(
                        learning_rate=lr,
                        decay_steps=decay_steps,
                        decay_rate=decay_rate,
                        staircase=True))

            # async sync heter
            if sync_mode in ["async", "sync", "heter"]:
                print("decay_steps: {}".format(decay_steps))
                scheduler = paddle.optimizer.lr.ExponentialDecay(
                    learning_rate=lr, gamma=decay_rate, verbose=True)
                optimizer = paddle.optimizer.SGD(scheduler)
                strategy.a_sync_configs = {"lr_decay_steps": decay_steps}

            optimizer = fleet.distributed_optimizer(optimizer, strategy)

        if pure_bf16:
            optimizer = paddle.static.amp.bf16.decorate_bf16(
                optimizer, use_bf16_guard=False, use_pure_bf16=pure_bf16)

        self.optimizer = optimizer
        self.optimizer.minimize(self.cost)
