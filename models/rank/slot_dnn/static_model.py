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
import paddle.fluid as fluid

from net import BenchmarkDNNLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.infer_target_var = None
        self.config = config
        self._init_hyper_parameters()
        self.sync_mode = config.get("runner.sync_mode")

    def _init_hyper_parameters(self):
        self.is_distributed = False
        self.distributed_embedding = False

        if self.config.get("hyper_parameters.distributed_embedding", 0) == 1:
            self.distributed_embedding = True

        self.dict_dim = self.config.get("hyper_parameters.dict_dim")
        self.emb_dim = self.config.get("hyper_parameters.emb_dim")
        self.slot_num = self.config.get("hyper_parameters.slot_num")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.layer_sizes = self.config.get("hyper_parameters.layer_sizes")

    def create_feeds(self, is_infer=False):

        slot_ids = [
            paddle.static.data(
                name=str(i), shape=[None, 1], dtype="int64", lod_level=1)
            for i in range(2, self.slot_num + 2)
        ]

        label = paddle.static.data(
            name="click", shape=[None, 1], dtype="int64", lod_level=1)

        feeds_list = [label] + slot_ids
        return feeds_list

    def net(self, input, is_infer=False):
        self.label_input = input[0]
        self.slot_inputs = input[1:]

        dnn_model = BenchmarkDNNLayer(
            self.dict_dim,
            self.emb_dim,
            self.slot_num,
            self.layer_sizes,
            sync_mode=self.sync_mode)

        self.predict = dnn_model(self.slot_inputs)

        # self.all_vars = input + dnn_model.all_vars
        self.all_vars = dnn_model.all_vars

        predict_2d = paddle.concat(x=[1 - self.predict, self.predict], axis=1)
        #label_int = paddle.cast(self.label, 'int64')

        auc, batch_auc_var, self.auc_stat_list = paddle.static.auc(
            input=predict_2d, label=self.label_input, slide_steps=0)
        self.metric_list = fluid.contrib.layers.ctr_metric_bundle(
            self.predict,
            fluid.layers.cast(
                x=self.label_input, dtype='float32'))
        self.inference_model_feed_vars = dnn_model.inference_model_feed_vars
        self.inference_target_var = self.predict

        if is_infer:
            fetch_dict = {'auc': auc}
            return fetch_dict
        cost = paddle.nn.functional.log_loss(
            input=self.predict, label=paddle.cast(self.label_input, "float32"))
        avg_cost = paddle.sum(x=cost)
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
