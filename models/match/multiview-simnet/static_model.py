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
from net import MultiviewSimnetLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.query_encoder = self.config.get("hyper_parameters.query_encoder")
        self.title_encoder = self.config.get("hyper_parameters.title_encoder")
        self.query_encode_dim = self.config.get(
            "hyper_parameters.query_encode_dim")
        self.title_encode_dim = self.config.get(
            "hyper_parameters.title_encode_dim")

        self.emb_size = self.config.get("hyper_parameters.sparse_feature_dim")
        self.emb_dim = self.config.get("hyper_parameters.embedding_dim")
        self.emb_shape = [self.emb_size, self.emb_dim]

        self.hidden_size = self.config.get("hyper_parameters.hidden_size")
        self.margin = self.config.get("hyper_parameters.margin")
        self.query_len = self.config.get("hyper_parameters.query_len")
        self.pos_len = self.config.get("hyper_parameters.pos_len")
        self.neg_len = self.config.get("hyper_parameters.neg_len")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")

    def create_feeds(self, is_infer=False):
        self.q_slots = paddle.static.data(
            name="q_slots", shape=[None, self.query_len], dtype='int64')

        self.pt_slots = paddle.static.data(
            name="pt_slots", shape=[None, self.pos_len], dtype='int64')

        if is_infer:
            feeds_list = [self.q_slots, self.pt_slots]
            return feeds_list

        self.nt_slots = paddle.static.data(
            name="nt_slots", shape=[None, self.neg_len], dtype="int64")
        feeds_list = [self.q_slots, self.pt_slots, self.nt_slots]

        return feeds_list

    def net(self, input, is_infer=False):
        self.q_slots = [input[0]]
        self.pt_slots = [input[1]]
        if not is_infer:
            self.batch_size = self.config.get("runner.train_batch_size")
            self.nt_slots = [input[2]]
            inputs = [self.q_slots, self.pt_slots, self.nt_slots]
        else:
            self.batch_size = self.config.get("runner.infer_batch_size")
            inputs = [self.q_slots, self.pt_slots]
        simnet_model = MultiviewSimnetLayer(
            self.query_encoder, self.title_encoder, self.query_encode_dim,
            self.title_encode_dim, self.emb_size, self.emb_dim,
            self.hidden_size, self.margin, self.query_len, self.pos_len,
            self.neg_len)
        cos_pos, cos_neg = simnet_model.forward(inputs, is_infer)

        self.inference_target_var = cos_pos
        if is_infer:
            fetch_dict = {'query_pt_sim': cos_pos}
            return fetch_dict
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
        self._cost = avg_cost
        self.acc = self.get_acc(cos_neg, cos_pos)
        fetch_dict = {'Acc': self.acc, 'Loss': avg_cost}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adam(
            learning_rate=self.learning_rate, lazy_mode=True)
        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)

    def get_acc(self, x, y):
        less = paddle.cast(paddle.less_than(x, y), dtype='float32')
        label_ones = paddle.full(
            dtype='float32', shape=[self.batch_size, 1], fill_value=1.0)
        correct = paddle.sum(less)
        total = paddle.sum(label_ones)
        acc = paddle.divide(correct, total)
        return acc
