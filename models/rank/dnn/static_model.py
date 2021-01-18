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
import paddle.nn.functional as F
import paddle.nn as nn
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
import math
from net import DNNLayer


class Model(object):
    """
    DNN for Click-Through Rate prediction
    """

    def __init__(self, config):
        self.cost = None
        self.metrics = {}
        self.config = config
        self.init_hyper_parameters()

    def init_hyper_parameters(self):
        self.dense_feature_dim = self.config.get(
            "hyper_parameters.dense_feature_dim")
        self.sparse_feature_dim = self.config.get(
            "hyper_parameters.sparse_feature_dim")
        self.embedding_size = self.config.get(
            "hyper_parameters.embedding_size")
        self.fc_sizes = self.config.get("hyper_parameters.fc_sizes")

        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.adam_lazy_mode = self.config.get(
            "hyper_parameters.optimizer.adam_lazy_mode")

    def input_data(self):
        dense_input = fluid.layers.data(
            name="dense_input",
            shape=[self.dense_feature_dim],
            dtype="float32")

        sparse_input_ids = [
            fluid.layers.data(
                name="C" + str(i), shape=[1], lod_level=1, dtype="int64")
            for i in range(1, 27)
        ]

        label = fluid.layers.data(name="label", shape=[1], dtype="int64")

        inputs = [dense_input] + sparse_input_ids + [label]
        return inputs

    def net(self, input):
        "Dynamic network -> Static network"
        dnn_model = DNNLayer(self.sparse_feature_dim, self.embedding_size,
                             self.dense_feature_dim,
                             len(input[1:-1]), self.fc_sizes)

        raw_predict_2d = dnn_model(input[1:-1], input[0])

        with fluid.device_guard("gpu"):
            predict_2d = paddle.nn.functional.softmax(raw_predict_2d)

            self.predict = predict_2d

            auc, batch_auc, _ = paddle.fluid.layers.auc(input=self.predict,
                                                        label=input[-1],
                                                        num_thresholds=2**12,
                                                        slide_steps=20)

            cost = paddle.nn.functional.cross_entropy(
                input=raw_predict_2d, label=input[-1])
            avg_cost = paddle.mean(x=cost)
            self.cost = avg_cost
            self.infer_target_var = auc

            sync_mode = self.config.get("static_benchmark.sync_mode")
            if sync_mode == "heter":
                fluid.layers.Print(auc, message="AUC")

        return {'cost': avg_cost, 'auc': auc}

    def minimize(self, strategy=None):
        optimizer = fluid.optimizer.Adam(
            self.learning_rate, lazy_mode=self.adam_lazy_mode)
        if strategy != None:
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self.cost)
