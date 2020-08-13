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

import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.feature_size = envs.get_global_env(
            "hyper_parameters.feature_size")
        self.bottom_size = envs.get_global_env("hyper_parameters.bottom_size")
        self.tower_size = envs.get_global_env("hyper_parameters.tower_size")
        self.tower_nums = envs.get_global_env("hyper_parameters.tower_nums")

    def input_data(self, is_infer=False, **kwargs):
        inputs = fluid.data(
            name="input", shape=[-1, self.feature_size], dtype="float32")
        label_income = fluid.data(
            name="label_income", shape=[-1, 2], dtype="float32", lod_level=0)
        label_marital = fluid.data(
            name="label_marital", shape=[-1, 2], dtype="float32", lod_level=0)
        if is_infer:
            return [inputs, label_income, label_marital]
        else:
            return [inputs, label_income, label_marital]

    def net(self, inputs, is_infer=False):
        input_data = inputs[0]
        label_income = inputs[1]
        label_marital = inputs[2]

        bottom_output = fluid.layers.fc(
            input=input_data,
            size=self.bottom_size,
            act='relu',
            bias_attr=fluid.ParamAttr(learning_rate=1.0),
            name='bottom_output')

        # Build tower layer from bottom layer
        output_layers = []
        for index in range(self.tower_nums):
            tower_layer = fluid.layers.fc(input=bottom_output,
                                          size=self.tower_size,
                                          act='relu',
                                          name='task_layer_' + str(index))
            output_layer = fluid.layers.fc(input=tower_layer,
                                           size=2,
                                           act='softmax',
                                           name='output_layer_' + str(index))
            output_layers.append(output_layer)

        pred_income = fluid.layers.clip(
            output_layers[0], min=1e-15, max=1.0 - 1e-15)
        pred_marital = fluid.layers.clip(
            output_layers[1], min=1e-15, max=1.0 - 1e-15)

        label_income_1 = fluid.layers.slice(
            label_income, axes=[1], starts=[1], ends=[2])
        label_marital_1 = fluid.layers.slice(
            label_marital, axes=[1], starts=[1], ends=[2])

        auc_income, batch_auc_1, auc_states_1 = fluid.layers.auc(
            input=pred_income,
            label=fluid.layers.cast(
                x=label_income_1, dtype='int64'))
        auc_marital, batch_auc_2, auc_states_2 = fluid.layers.auc(
            input=pred_marital,
            label=fluid.layers.cast(
                x=label_marital_1, dtype='int64'))

        if is_infer:
            self._infer_results["AUC_income"] = auc_income
            self._infer_results["AUC_marital"] = auc_marital
            return

        cost_income = fluid.layers.cross_entropy(
            input=pred_income, label=label_income, soft_label=True)
        cost_marital = fluid.layers.cross_entropy(
            input=pred_marital, label=label_marital, soft_label=True)
        cost = fluid.layers.elementwise_add(cost_income, cost_marital, axis=1)

        avg_cost = fluid.layers.mean(x=cost)

        self._cost = avg_cost
        self._metrics["AUC_income"] = auc_income
        self._metrics["BATCH_AUC_income"] = batch_auc_1
        self._metrics["AUC_marital"] = auc_marital
        self._metrics["BATCH_AUC_marital"] = batch_auc_2
