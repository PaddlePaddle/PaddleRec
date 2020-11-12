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
import paddle


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
        inputs = paddle.fluid.data(
            name="input", shape=[-1, self.feature_size], dtype="float32")
        label_income = paddle.fluid.data(
            name="label_income", shape=[-1, 2], dtype="float32", lod_level=0)
        label_marital = paddle.fluid.data(
            name="label_marital", shape=[-1, 2], dtype="float32", lod_level=0)
        if is_infer:
            return [inputs, label_income, label_marital]
        else:
            return [inputs, label_income, label_marital]

    def net(self, inputs, is_infer=False):
        input_data = inputs[0]
        label_income = inputs[1]
        label_marital = inputs[2]

        bottom_output = paddle.static.nn.fc(
            x=input_data,
            size=self.bottom_size,
            activation='relu',
            bias_attr=paddle.ParamAttr(learning_rate=1.0),
            name='bottom_output')

        # Build tower layer from bottom layer
        output_layers = []
        for index in range(self.tower_nums):
            tower_layer = paddle.static.nn.fc(x=bottom_output,
                                              size=self.tower_size,
                                              activation='relu',
                                              name='task_layer_' + str(index))
            output_layer = paddle.static.nn.fc(
                x=tower_layer,
                size=2,
                activation='softmax',
                name='output_layer_' + str(index))
            output_layers.append(output_layer)

        pred_income = paddle.fluid.layers.clip(
            output_layers[0], min=1e-15, max=1.0 - 1e-15)
        pred_income_1 = paddle.slice(
            pred_income, axes=[1], starts=[0], ends=[1])
        pred_marital = paddle.fluid.layers.clip(
            output_layers[1], min=1e-15, max=1.0 - 1e-15)
        pred_marital_1 = paddle.slice(
            pred_marital, axes=[1], starts=[0], ends=[1])

        label_income_1 = paddle.slice(
            label_income, axes=[1], starts=[1], ends=[2])
        label_marital_1 = paddle.slice(
            label_marital, axes=[1], starts=[1], ends=[2])

        auc_income, batch_auc_1, auc_states_1 = paddle.fluid.layers.auc(
            input=pred_income,
            label=paddle.cast(
                x=label_income_1, dtype='int64'))
        auc_marital, batch_auc_2, auc_states_2 = paddle.fluid.layers.auc(
            input=pred_marital,
            label=paddle.cast(
                x=label_marital_1, dtype='int64'))
        if is_infer:
            self._infer_results["AUC_income"] = auc_income
            self._infer_results["AUC_marital"] = auc_marital
            return

        cost_income = paddle.nn.functional.log_loss(
            input=pred_income_1, label=label_income_1)
        cost_marital = paddle.nn.functional.log_loss(
            input=pred_marital_1, label=label_marital_1)

        avg_cost_income = paddle.mean(x=cost_income)
        avg_cost_marital = paddle.mean(x=cost_marital)

        cost = avg_cost_income + avg_cost_marital

        self._cost = cost
        self._metrics["AUC_income"] = auc_income
        self._metrics["BATCH_AUC_income"] = batch_auc_1
        self._metrics["AUC_marital"] = auc_marital
        self._metrics["BATCH_AUC_marital"] = batch_auc_2
