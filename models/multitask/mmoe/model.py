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
        self.expert_num = envs.get_global_env("hyper_parameters.expert_num")
        self.gate_num = envs.get_global_env("hyper_parameters.gate_num")
        self.expert_size = envs.get_global_env("hyper_parameters.expert_size")
        self.tower_size = envs.get_global_env("hyper_parameters.tower_size")

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

        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper
        expert_outputs = []
        for i in range(0, self.expert_num):
            expert_output = paddle.static.nn.fc(
                x=input_data,
                size=self.expert_size,
                activation='relu',
                bias_attr=paddle.ParamAttr(learning_rate=1.0),
                name='expert_' + str(i))
            expert_outputs.append(expert_output)
        expert_concat = paddle.concat(x=expert_outputs, axis=1)
        #expert_concat = paddle.fluid.layers.nn.reshape(
        expert_concat = paddle.reshape(
            expert_concat, [-1, self.expert_num, self.expert_size])

        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper
        output_layers = []
        for i in range(0, self.gate_num):
            cur_gate = paddle.static.nn.fc(
                x=input_data,
                size=self.expert_num,
                activation='softmax',
                bias_attr=paddle.ParamAttr(learning_rate=1.0),
                name='gate_' + str(i))
            # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
            cur_gate_expert = paddle.multiply(
                x=expert_concat, y=cur_gate, axis=0)
            cur_gate_expert = paddle.sum(x=cur_gate_expert, axis=1)
            # Build tower layer
            cur_tower = paddle.static.nn.fc(x=cur_gate_expert,
                                            size=self.tower_size,
                                            activation='relu',
                                            name='task_layer_' + str(i))
            out = paddle.static.nn.fc(x=cur_tower,
                                      size=2,
                                      activation='softmax',
                                      name='out_' + str(i))

            output_layers.append(out)

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

        cost_income = paddle.fluid.layers.cross_entropy(
            input=pred_income, label=label_income, soft_label=True)
        cost_marital = paddle.fluid.layers.cross_entropy(
            input=pred_marital, label=label_marital, soft_label=True)

        cost_income = paddle.nn.functional.log_loss(
            input=pred_income_1, label=label_income_1)
        cost_marital = paddle.nn.functional.log_loss(
            input=pred_marital_1, label=label_marital_1)
        #cost_income = paddle.fluid.layers.cross_entropy(
        #    input=pred_income, label=label_income, soft_label=True)
        #cost_marital = paddle.fluid.layers.cross_entropy(
        #    input=pred_marital, label=label_marital, soft_label=True)
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
