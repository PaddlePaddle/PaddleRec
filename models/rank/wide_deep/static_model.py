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
import math
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from wide_deep_net import WideDeepLayer


class Model(object):
    def __init__(self, config):
        self.cost = None
        self.metrics = {}
        self.config = config
        self.init_hyper_parameters()

    def init_hyper_parameters(self):
        self.sparse_feature_number = self.config.get(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = self.config.get(
            "hyper_parameters.sparse_feature_dim")
        self.sparse_inputs_slot = self.config.get(
            "hyper_parameters.sparse_inputs_slots")
        self.dense_input_dim = self.config.get(
            "hyper_parameters.dense_input_dim")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.fc_sizes = self.config.get("hyper_parameters.fc_sizes")
        self.adam_lazy_mode = self.config.get(
            "hyper_parameters.optimizer.adam_lazy_mode")

    def input_data(self):
        dense_input = fluid.layers.data(
            name="dense_input", shape=[self.dense_input_dim], dtype="float32")

        sparse_input_ids = [
            fluid.layers.data(
                name="C" + str(i), shape=[1], lod_level=1, dtype="int64")
            for i in range(1, self.sparse_inputs_slot)
        ]

        label = fluid.layers.data(name="label", shape=[1], dtype="int64")

        inputs = [dense_input] + sparse_input_ids + [label]
        return inputs

    def net(self, inputs, is_infer=False):
        self.sparse_inputs = inputs[1:-1]
        self.dense_input = inputs[0]
        self.label_input = inputs[-1]

        wide_deep_model = WideDeepLayer(
            self.sparse_feature_number, self.sparse_feature_dim,
            self.dense_input_dim, self.sparse_inputs_slot - 1, self.fc_sizes)

        pred = wide_deep_model(self.sparse_inputs, self.dense_input)

        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)

        auc, batch_auc, _ = paddle.fluid.layers.auc(input=predict_2d,
                                                    label=self.label_input,
                                                    num_thresholds=2**12,
                                                    slide_steps=20)

        if is_infer:
            self._infer_results["AUC"] = auc
            return

        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                self.label_input, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        self.cost = avg_cost
        self.infer_target_var = auc
        return {'cost': avg_cost, 'auc': auc}

    def minimize(self, strategy=None):
        optimizer = paddle.optimizer.Adam(
            self.learning_rate, lazy_mode=self.adam_lazy_mode)
        if strategy != None:
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self.cost)

    def infer_net(self):
        pass
