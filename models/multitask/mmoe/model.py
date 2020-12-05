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

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase
from paddlerec.core.metrics import AUC
import paddle
from mmoe_net import MMoELayer


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.feature_size = envs.get_global_env(
            "hyper_parameters.feature_size")
        self.expert_num = envs.get_global_env("hyper_parameters.expert_num")
        self.gate_num = envs.get_global_env("hyper_parameters.gate_num")
        self.expert_size = envs.get_global_env("hyper_parameters.expert_size")
        self.tower_size = envs.get_global_env("hyper_parameters.tower_size")

    def input_data(self, is_infer=False, **kwargs):
        inputs = paddle.static.data(
            name="input", shape=[-1, self.feature_size], dtype="float32")
        label_income = paddle.static.data(
            name="label_income", shape=[-1, 1], dtype="float32", lod_level=0)
        label_marital = paddle.static.data(
            name="label_marital", shape=[-1, 1], dtype="float32", lod_level=0)
        if is_infer:
            return [inputs, label_income, label_marital]
        else:
            return [inputs, label_income, label_marital]

    def net(self, inputs, is_infer=False):
        input_data = inputs[0]
        label_income = inputs[1]
        label_marital = inputs[2]

        MMoE = MMoELayer(self.feature_size, self.expert_num, self.expert_size,
                         self.tower_size, self.gate_num)
        pred_income, pred_marital = MMoE(input_data)

        pred_income_1 = paddle.slice(
            pred_income, axes=[1], starts=[1], ends=[2])
        pred_marital_1 = paddle.slice(
            pred_marital, axes=[1], starts=[1], ends=[2])

        auc_income, batch_auc_1, auc_states_1 = paddle.fluid.layers.auc(
            #auc_income = AUC(
            input=pred_income,
            label=paddle.cast(
                x=label_income, dtype='int64'))
        #auc_marital = AUC(
        auc_marital, batch_auc_2, auc_states_2 = paddle.fluid.layers.auc(
            input=pred_marital,
            label=paddle.cast(
                x=label_marital, dtype='int64'))
        if is_infer:
            self._infer_results["AUC_income"] = auc_income
            self._infer_results["AUC_marital"] = auc_marital
            return
        # 1.8 cross_entropy
        cost_income = paddle.nn.functional.log_loss(
            input=pred_income_1, label=label_income)
        cost_marital = paddle.nn.functional.log_loss(
            input=pred_marital_1, label=label_marital)

        avg_cost_income = paddle.mean(x=cost_income)
        avg_cost_marital = paddle.mean(x=cost_marital)

        cost = avg_cost_income + avg_cost_marital

        self._cost = cost
        self._metrics["AUC_income"] = auc_income
        self._metrics["BATCH_AUC_income"] = batch_auc_1
        self._metrics["AUC_marital"] = auc_marital
        self._metrics["BATCH_AUC_marital"] = batch_auc_2

    def infer_net(self):
        pass
