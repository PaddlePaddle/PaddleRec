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

from net import Word2VecLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
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

    def create_feeds(self, is_infer=False):
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

        true_logits = paddle.nn.functional.sigmoid(true_logits)
        true_xent = paddle.nn.functional.binary_cross_entropy(true_logits,
                                                              label_ones)
        neg_logits = paddle.nn.functional.sigmoid(neg_logits)
        neg_xent = paddle.nn.functional.binary_cross_entropy(neg_logits,
                                                             label_zeros)
        cost = paddle.add(true_xent, neg_xent)
        avg_cost = paddle.mean(x=cost)

        self._cost = avg_cost
        fetch_dict = {'loss': avg_cost}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.SGD(learning_rate=self.learning_rate)
        #            learning_rate=paddle.fluid.layers.exponential_decay(
        #                learning_rate=self.learning_rate,
        #                decay_steps=self.decay_steps,
        #                decay_rate=self.decay_rate,
        #                staircase=True))
        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        return optimizer

    def infer_net(self, inputs):
        [analogy_a, analogy_b, analogy_c] = inputs

        all_label = paddle.static.data(
            name="all_label", shape=[vocab_size], dtype='int64')
        word2vec = net.Word2VecInferLayer(self.sparse_feature_number,
                                          self.sparse_feature_dim, "emb")
        word2vec.forward(input[0], input[1], input[2], input[3], all_label)
        val, pred_idx = word2vec.forward(inputs[0], inputs[1], inputs[2],
                                         inputs[3], all_label)
        return val, pred_idx
