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
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import math
import config
from ...models.recall.word2vec.word2vec_net import Word2VecLayer


class Model(object):
    """
    DNN for Click-Through Rate prediction
    """

    def input_var(self):
        input_word = paddle.static.data(
            name="input_word", shape=[None, 1], dtype='int64')
        true_word = paddle.static.data(
            name='true_label', shape=[None, 1], dtype='int64')
        if config.with_shuffle_batch:
            return [input_word, true_word]

        neg_word = paddle.static.data(
            name="neg_label", shape=[None, config.neg_num], dtype='int64')
        return [input_word, true_word, neg_word]

    def train_net(self, inputs):
        word2vec_model = Word2VecLayer(
            config.sparse_feature_number,
            config.sparse_feature_dim,
            config.neg_num,
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

        self.cost = avg_cost

        return {'loss': avg_cost}

    def minimize(self, strategy=None):
        scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=config.learning_rate,
            gamma=config.decay_rate,
            verbose=True)
        optimizer = fluid.optimizer.SGD(scheduler)
        if strategy != None:
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self.cost)
