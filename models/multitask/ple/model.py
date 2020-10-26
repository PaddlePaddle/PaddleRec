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
        self.task_num = envs.get_global_env("hyper_parameters.task_num")
        self.exp_per_task = envs.get_global_env(
            "hyper_parameters.exp_per_task")
        self.shared_num = envs.get_global_env("hyper_parameters.shared_num")
        self.gate_num = envs.get_global_env("hyper_parameters.gate_num")
        self.expert_size = envs.get_global_env("hyper_parameters.expert_size")
        self.tower_size = envs.get_global_env("hyper_parameters.tower_size")
        self.level_number = envs.get_global_env(
            "hyper_parameters.level_number")

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

    def ple_net(self, inputs, if_last, level_name):
        # inputs: [input_task1, input_task2 ... input_taskn, shared_input]
        expert_outputs = []

        # task-specific expert part
        for i in range(0, self.task_num):
            for j in range(0, self.exp_per_task):
                expert_output = fluid.layers.fc(
                    input=inputs[i],
                    size=self.expert_size,
                    act='relu',
                    bias_attr=fluid.ParamAttr(learning_rate=1.0),
                    name=level_name + "_exp_" + str(i) + "_" + str(j))
                expert_outputs.append(expert_output)
        # shared expert part
        for i in range(0, self.shared_num):
            expert_output = fluid.layers.fc(
                input=inputs[-1],
                size=self.expert_size,
                act='relu',
                bias_attr=fluid.ParamAttr(learning_rate=1.0),
                name=level_name + "_exp_shared_" + str(i))
            expert_outputs.append(expert_output)

        # task gate part
        outputs = []
        for i in range(0, self.task_num):
            cur_expert_num = self.exp_per_task + self.shared_num
            cur_gate = fluid.layers.fc(
                input=inputs[i],
                size=cur_expert_num,
                act='softmax',
                bias_attr=fluid.ParamAttr(learning_rate=1.0),
                name=level_name + "_gate_" + str(i))
            # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x)) 
            cur_experts = expert_outputs[i * self.exp_per_task:(
                i + 1) * self.exp_per_task] + expert_outputs[-int(
                    self.task_num):]
            expert_concat = fluid.layers.concat(cur_experts, axis=1)
            expert_concat = fluid.layers.reshape(
                expert_concat, [-1, cur_expert_num, self.expert_size])

            cur_gate_expert = fluid.layers.elementwise_mul(
                expert_concat, cur_gate, axis=0)
            cur_gate_expert = fluid.layers.reduce_sum(cur_gate_expert, dim=1)
            outputs.append(cur_gate_expert)

        # shared gate
        if not if_last:
            cur_expert_num = self.task_num * self.exp_per_task + self.shared_num
            cur_gate = fluid.layers.fc(
                input=inputs[-1],
                size=cur_expert_num,
                act='softmax',
                bias_attr=fluid.ParamAttr(learning_rate=1.0),
                name=level_name + "_gate_shared")
            # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x)) 
            cur_experts = expert_outputs
            expert_concat = fluid.layers.concat(cur_experts, axis=1)
            expert_concat = fluid.layers.reshape(
                expert_concat, [-1, cur_expert_num, self.expert_size])

            cur_gate_expert = fluid.layers.elementwise_mul(
                expert_concat, cur_gate, axis=0)
            cur_gate_expert = fluid.layers.reduce_sum(cur_gate_expert, dim=1)
            outputs.append(cur_gate_expert)

        return outputs

    def net(self, inputs, is_infer=False):
        input_data = inputs[0]
        inputs_ple = []
        for i in range(0, self.task_num + 1):
            inputs_ple.append(input_data)
        label_income = inputs[1]
        label_marital = inputs[2]

        outputs_ple = []
        for i in range(0, self.level_number):
            if i == self.level_number - 1:
                outputs_ple = self.ple_net(
                    inputs=inputs_ple,
                    if_last=True,
                    level_name=("lev" + str(i)))
                break
            else:
                outputs_ple = self.ple_net(
                    inputs=inputs_ple,
                    if_last=False,
                    level_name=("lev" + str(i)))
                inputs_ple = outputs_ple

        output_layers = []
        for i in range(0, self.task_num):
            # Build tower layer
            cur_tower = fluid.layers.fc(input=outputs_ple[i],
                                        size=self.tower_size,
                                        act='relu',
                                        name='task_layer_' + str(i))
            out = fluid.layers.fc(input=cur_tower,
                                  size=2,
                                  act='softmax',
                                  name='out_' + str(i))

            output_layers.append(out)

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

        avg_cost_income = fluid.layers.mean(x=cost_income)
        avg_cost_marital = fluid.layers.mean(x=cost_marital)

        cost = avg_cost_income + avg_cost_marital

        self._cost = cost
        self._metrics["AUC_income"] = auc_income
        self._metrics["BATCH_AUC_income"] = batch_auc_1
        self._metrics["AUC_marital"] = auc_marital
        self._metrics["BATCH_AUC_marital"] = batch_auc_2

    def infer_net(self):
        pass
