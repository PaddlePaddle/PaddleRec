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

from net import WideDeepLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()
        self.sync_mode = config.get("runner.sync_mode")

    def _init_hyper_parameters(self):
        self.is_distributed = False
        self.distributed_embedding = False

        if self.config.get("hyper_parameters.distributed_embedding", 0) == 1:
            self.distributed_embedding = True

        self.sparse_feature_number = self.config.get(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = self.config.get(
            "hyper_parameters.sparse_feature_dim")
        self.sparse_inputs_slots = self.config.get(
            "hyper_parameters.sparse_inputs_slots")
        self.dense_input_dim = self.config.get(
            "hyper_parameters.dense_input_dim")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.fc_sizes = self.config.get("hyper_parameters.fc_sizes")

    def create_feeds(self, is_infer=False):
        dense_input = paddle.static.data(
            name="dense_input",
            shape=[None, self.dense_input_dim],
            dtype="float32")

        # sparse_input_ids = [
        #     paddle.static.data(
        #         name="C" + str(i), shape=[None, 1], dtype="int64")
        #     for i in range(1, self.sparse_inputs_slots)
        # ]

        sparse_input_ids = [
            paddle.static.data(
                name=str(i), shape=[None, 1], dtype="int64")
            for i in range(1, self.sparse_inputs_slots)
        ]

        label = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64")

        self._sparse_data_var = [label] + sparse_input_ids
        self._dense_data_var = [dense_input]

        feeds_list = [label] + sparse_input_ids + [dense_input]
        return feeds_list

    def net(self, input, is_infer=False):
        self.label_input = input[0]
        self.sparse_inputs = input[1:self.sparse_inputs_slots]
        self.dense_input = input[-1]
        sparse_number = self.sparse_inputs_slots - 1

        wide_deep_model = WideDeepLayer(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            self.dense_input_dim,
            sparse_number,
            self.fc_sizes,
            sync_mode=self.sync_mode)

        self.cast_label = paddle.cast(self.label_input, dtype='float32')
        ones = paddle.full_like(
            self.label_input, fill_value=1, dtype="float32")
        show_click = paddle.cast(
            paddle.concat(
                [ones, self.cast_label], axis=1), dtype='float32')
        show_click.stop_gradient = True

        pred = wide_deep_model.forward(self.sparse_inputs, self.dense_input,
                                       show_click)

        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)

        self.predict = predict_2d

        auc, batch_auc, [
            self.batch_stat_pos, self.batch_stat_neg, self.stat_pos,
            self.stat_neg
        ] = paddle.static.auc(input=self.predict,
                              label=self.label_input,
                              num_thresholds=2**12,
                              slide_steps=20)
        auc = paddle.cast(auc, "float32")
        self.inference_target_var = auc
        if is_infer:
            fetch_dict = {'auc': auc}
            return fetch_dict

        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                self.label_input, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        self._cost = avg_cost

        fetch_dict = {'cost': avg_cost, 'auc': auc}
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
