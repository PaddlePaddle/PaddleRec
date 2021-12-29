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

from net import PLELayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.feature_size = self.config.get("hyper_parameters.feature_size")
        self.task_num = self.config.get("hyper_parameters.task_num")
        self.exp_per_task = self.config.get("hyper_parameters.exp_per_task")
        self.shared_num = self.config.get("hyper_parameters.shared_num")
        self.expert_size = self.config.get("hyper_parameters.expert_size")
        self.tower_size = self.config.get("hyper_parameters.tower_size")
        self.level_number = self.config.get("hyper_parameters.level_number")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")

    def create_feeds(self, is_infer=False):
        inputs = paddle.static.data(
            name="input", shape=[-1, self.feature_size], dtype="float32")
        label_income = paddle.static.data(
            name="label_income", shape=[-1, 1], dtype="int64", lod_level=0)
        label_marital = paddle.static.data(
            name="label_marital", shape=[-1, 1], dtype="int64", lod_level=0)
        if is_infer:
            return [inputs, label_income, label_marital]
        else:
            return [inputs, label_income, label_marital]

    def net(self, inputs, is_infer=False):
        input_data = inputs[0]
        label_income = inputs[1]
        label_marital = inputs[2]

        PLE = PLELayer(self.feature_size, self.task_num, self.exp_per_task,
                       self.shared_num, self.expert_size, self.tower_size,
                       self.level_number)
        pred_income, pred_marital = PLE.forward(input_data)

        pred_income_1 = paddle.slice(
            pred_income, axes=[1], starts=[1], ends=[2])
        pred_marital_1 = paddle.slice(
            pred_marital, axes=[1], starts=[1], ends=[2])

        auc_income, batch_auc_1, auc_states_1 = paddle.static.auc(
            #auc_income = AUC(
            input=pred_income,
            label=paddle.cast(
                x=label_income, dtype='int64'))
        #auc_marital = AUC(
        auc_marital, batch_auc_2, auc_states_2 = paddle.static.auc(
            input=pred_marital,
            label=paddle.cast(
                x=label_marital, dtype='int64'))
        if is_infer:
            fetch_dict = {'auc_income': auc_income, 'auc_marital': auc_marital}
            return fetch_dict
        cost_income = paddle.nn.functional.log_loss(
            input=pred_income_1,
            label=paddle.cast(
                label_income, dtype="float32"))
        cost_marital = paddle.nn.functional.log_loss(
            input=pred_marital_1,
            label=paddle.cast(
                label_marital, dtype="float32"))

        avg_cost_income = paddle.mean(x=cost_income)
        avg_cost_marital = paddle.mean(x=cost_marital)

        cost = avg_cost_income + avg_cost_marital

        self._cost = cost
        fetch_dict = {
            'cost': cost,
            'auc_income': auc_income,
            'auc_marital': auc_marital
        }
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
