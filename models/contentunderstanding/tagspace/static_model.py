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

from net import TagspaceLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.vocab_text_size = self.config.get(
            "hyper_parameters.vocab_text_size")
        self.vocab_tag_size = self.config.get(
            "hyper_parameters.vocab_tag_size")
        self.emb_dim = self.config.get("hyper_parameters.emb_dim")
        self.hid_dim = self.config.get("hyper_parameters.hid_dim")
        self.win_size = self.config.get("hyper_parameters.win_size")
        self.margin = self.config.get("hyper_parameters.margin")
        self.neg_size = self.config.get("hyper_parameters.neg_size")
        self.text_len = self.config.get("hyper_parameters.text_len")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")

    def create_feeds(self, is_infer=False):
        text = paddle.static.data(
            name="text", shape=[None, self.text_len], dtype='int64')
        pos_tag = paddle.static.data(
            name="pos_tag", shape=[None, 1], dtype='int64')
        neg_tag = paddle.static.data(
            name="neg_tag", shape=[None, self.neg_size], dtype='int64')
        feeds_list = [text, pos_tag, neg_tag]
        return feeds_list

    def net(self, input, is_infer=False):
        if is_infer:
            self.batch_size = self.config.get("runner.infer_batch_size")
        else:
            self.batch_size = self.config.get("runner.train_batch_size")
        tagspace_model = TagspaceLayer(
            self.vocab_text_size, self.vocab_tag_size, self.emb_dim,
            self.hid_dim, self.win_size, self.margin, self.neg_size,
            self.text_len)
        cos_pos, cos_neg = tagspace_model.forward(input)
        # calculate hinge loss
        loss_part1 = paddle.subtract(
            paddle.full(
                shape=[self.batch_size, 1],
                fill_value=self.margin,
                dtype='float32'),
            cos_pos)
        loss_part2 = paddle.add(loss_part1, cos_neg)
        loss_part3 = paddle.maximum(
            paddle.full(
                shape=[self.batch_size, 1], fill_value=0.0, dtype='float32'),
            loss_part2)
        avg_cost = paddle.mean(loss_part3)

        less = paddle.cast(paddle.less_than(cos_neg, cos_pos), dtype='float32')
        label_ones = paddle.full(
            dtype='float32', shape=[self.batch_size, 1], fill_value=1.0)
        correct = paddle.sum(less)
        total = paddle.sum(label_ones)
        acc = paddle.divide(correct, total)
        self.inference_target_var = acc

        if is_infer:
            fetch_dict = {'ACC': acc}
            return fetch_dict

        self._cost = avg_cost

        fetch_dict = {'cost': avg_cost, 'ACC': acc}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adagrad(learning_rate=self.learning_rate)
        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)
