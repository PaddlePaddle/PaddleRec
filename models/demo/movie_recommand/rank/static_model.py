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
import paddle.nn.functional as F
import numpy as np
from net import DNNLayer


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
        self.hidden_layers = self.config.get("hyper_parameters.fc_sizes")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")

    def create_feeds(self, is_infer=False):
        userid = paddle.static.data(
            name="userid", shape=[-1, 1], dtype='int64')
        gender = paddle.static.data(
            name="gender", shape=[-1, 1], dtype='int64')
        age = paddle.static.data(name="age", shape=[-1, 1], dtype='int64')
        occupation = paddle.static.data(
            name="occupation", shape=[-1, 1], dtype='int64')
        user_sparse_inputs = [userid, gender, age, occupation]

        movieid = paddle.static.data(
            name="movieid", shape=[-1, 1], dtype='int64')
        title = paddle.static.data(name="title", shape=[-1, 4], dtype='int64')
        genres = paddle.static.data(
            name="genres", shape=[-1, 3], dtype='int64')
        mov_sparse_inputs = [movieid, title, genres]

        label_input = paddle.static.data(
            name="label", shape=[-1, 1], dtype='int64')

        feeds_list = user_sparse_inputs + mov_sparse_inputs + [label_input]
        return feeds_list

    def net(self, input, is_infer=False):
        self.user_sparse_inputs = input[:4]
        self.mov_sparse_inputs = input[4:7]
        self.label_input = input[-1]
        if is_infer:
            self.batch_size = self.config.get("runner.infer_batch_size")
        else:
            self.batch_size = self.config.get("runner.train_batch_size")
        rank_model = DNNLayer(self.sparse_feature_number,
                              self.sparse_feature_dim, self.hidden_layers)
        predict = rank_model.forward(self.batch_size, self.user_sparse_inputs,
                                     self.mov_sparse_inputs, self.label_input)

        self.inference_target_var = predict
        if is_infer:
            uid = self.user_sparse_inputs[0]
            movieid = self.mov_sparse_inputs[0]
            label = self.label_input
            predict = predict
            fetch_dict = {
                'userid': uid,
                'movieid': movieid,
                'label': label,
                'predict': predict
            }
            return fetch_dict
        cost = F.square_error_cost(
            predict, paddle.cast(
                x=self.label_input, dtype='float32'))
        avg_cost = paddle.mean(cost)
        self._cost = avg_cost
        fetch_dict = {'Loss': avg_cost}
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
