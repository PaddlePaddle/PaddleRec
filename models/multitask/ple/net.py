#the weight randly Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class PLELayer(nn.Layer):
    def __init__(self, feature_size, task_num, exp_per_task, shared_num,
                 expert_size, tower_size, level_number):
        super(PLELayer, self).__init__()

        self.task_num = task_num
        self.exp_per_task = exp_per_task
        self.shared_num = shared_num
        self.expert_size = expert_size
        self.tower_size = tower_size
        self.level_number = level_number

        # ple layer
        self.ple_layers = []
        for i in range(0, self.level_number):
            if i == self.level_number - 1:
                ple_layer = self.add_sublayer(
                    name='lev_' + str(i),
                    sublayer=SinglePLELayer(
                        feature_size, task_num, exp_per_task, shared_num,
                        expert_size, 'lev_' + str(i), True))
                self.ple_layers.append(ple_layer)
                break
            else:
                ple_layer = self.add_sublayer(
                    name='lev_' + str(i),
                    sublayer=SinglePLELayer(
                        feature_size, task_num, exp_per_task, shared_num,
                        expert_size, 'lev_' + str(i), False))
                self.ple_layers.append(ple_layer)
                feature_size = expert_size

        # task tower
        self._param_tower = []
        self._param_tower_out = []
        task_init = [pow(10, -i) for i in range(1, self.task_num + 1)]
        for i in range(0, self.task_num):
            linear = self.add_sublayer(
                name='tower_' + str(i),
                sublayer=nn.Linear(
                    expert_size,
                    tower_size,
                    #initialize each task respectly
                    weight_attr=nn.initializer.Constant(value=task_init[i]),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    #bias_attr=paddle.ParamAttr(learning_rate=1.0),
                    name='tower_' + str(i)))
            self._param_tower.append(linear)

            linear = self.add_sublayer(
                name='tower_out_' + str(i),
                sublayer=nn.Linear(
                    tower_size,
                    2,
                    #initialize each task respectly
                    weight_attr=nn.initializer.Constant(value=task_init[i]),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    name='tower_out_' + str(i)))
            self._param_tower_out.append(linear)

    def forward(self, input_data):
        inputs_ple = []
        # task_num part + shared part
        for i in range(0, self.task_num + 1):
            inputs_ple.append(input_data)
        # multiple ple layer
        ple_out = []
        for i in range(0, self.level_number):
            ple_out = self.ple_layers[i](inputs_ple)
            inputs_ple = ple_out

        #assert len(ple_out) == self.task_num
        output_layers = []
        for i in range(0, self.task_num):
            cur_tower = self._param_tower[i](ple_out[i])
            cur_tower = F.relu(cur_tower)
            out = self._param_tower_out[i](cur_tower)
            out = F.softmax(out)
            out = paddle.clip(out, min=1e-15, max=1.0 - 1e-15)
            output_layers.append(out)

        return output_layers


class SinglePLELayer(nn.Layer):
    def __init__(self, input_feature_size, task_num, exp_per_task, shared_num,
                 expert_size, level_name, if_last):
        super(SinglePLELayer, self).__init__()

        self.task_num = task_num
        self.exp_per_task = exp_per_task
        self.shared_num = shared_num
        self.expert_size = expert_size
        self.if_last = if_last

        self._param_expert = []
        # task-specific expert part
        step = self.exp_per_task
        for i in range(0, self.task_num):
            exp_init = [
                pow(10, -k) for k in range(1 + i * step, step * (i + 1) + 1)
            ]
            for j in range(0, self.exp_per_task):
                linear = self.add_sublayer(
                    name=level_name + "_exp_" + str(i) + "_" + str(j),
                    sublayer=nn.Linear(
                        input_feature_size,
                        expert_size,
                        #initialize each expert respectly
                        weight_attr=nn.initializer.Constant(value=exp_init[j]),
                        bias_attr=nn.initializer.Constant(value=0.1),
                        name=level_name + "_exp_" + str(i) + "_" + str(j)))
                self._param_expert.append(linear)
        shared_exp_init = [pow(10, -i) for i in range(1, self.shared_num + 1)]
        # shared expert part
        for i in range(0, self.shared_num):
            linear = self.add_sublayer(
                name=level_name + "_exp_shared_" + str(i),
                sublayer=nn.Linear(
                    input_feature_size,
                    expert_size,
                    #initialize each shared expert respectly  
                    weight_attr=nn.initializer.Constant(
                        value=shared_exp_init[i]),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    name=level_name + "_exp_shared_" + str(i)))
            self._param_expert.append(linear)

        # task gate part
        self._param_gate = []
        cur_expert_num = self.exp_per_task + self.shared_num
        gate_init = [pow(10, -i) for i in range(1, self.task_num + 1)]
        for i in range(0, self.task_num):
            linear = self.add_sublayer(
                name=level_name + "_gate_" + str(i),
                sublayer=nn.Linear(
                    input_feature_size,
                    cur_expert_num,
                    #initialize each gate respectly
                    weight_attr=nn.initializer.Constant(value=gate_init[i]),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    name=level_name + "_gate_" + str(i)))
            self._param_gate.append(linear)

        # shared gate
        if not if_last:
            cur_expert_num = self.task_num * self.exp_per_task + self.shared_num
            linear = self.add_sublayer(
                name=level_name + "_gate_shared_",
                sublayer=nn.Linear(
                    input_feature_size,
                    cur_expert_num,
                    weight_attr=nn.initializer.Constant(value=0.1),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    name=level_name + "_gate_shared_"))
            self._param_gate_shared = linear

    def forward(self, input_data):
        expert_outputs = []
        # task-specific expert part
        for i in range(0, self.task_num):
            for j in range(0, self.exp_per_task):
                linear_out = self._param_expert[i * self.exp_per_task + j](input_data[i])
                expert_output = F.relu(linear_out)
                expert_outputs.append(expert_output)
        # shared expert part
        for i in range(0, self.shared_num):
            linear_out = self._param_expert[self.exp_per_task * self.task_num +
                                            i](input_data[-1])
            expert_output = F.relu(linear_out)
            expert_outputs.append(expert_output)
        # task gate part
        outputs = []
        for i in range(0, self.task_num):
            cur_expert_num = self.exp_per_task + self.shared_num
            linear_out = self._param_gate[i](input_data[i])
            cur_gate = F.softmax(linear_out)
            cur_gate = paddle.reshape(cur_gate, [-1, cur_expert_num, 1])
            # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
            cur_experts = expert_outputs[i * self.exp_per_task:(
                i + 1) * self.exp_per_task] + expert_outputs[-int(
                    self.shared_num):]
            expert_concat = paddle.concat(x=cur_experts, axis=1)
            expert_concat = paddle.reshape(
                expert_concat, [-1, cur_expert_num, self.expert_size])
            cur_gate_expert = paddle.multiply(x=expert_concat, y=cur_gate)
            cur_gate_expert = paddle.sum(x=cur_gate_expert, axis=1)
            outputs.append(cur_gate_expert)

        # shared gate
        if not self.if_last:
            cur_expert_num = self.task_num * self.exp_per_task + self.shared_num
            linear_out = self._param_gate_shared(input_data[-1])
            cur_gate = F.softmax(linear_out)
            cur_gate = paddle.reshape(cur_gate, [-1, cur_expert_num, 1])
            cur_experts = expert_outputs
            expert_concat = paddle.concat(x=cur_experts, axis=1)
            expert_concat = paddle.reshape(
                expert_concat, [-1, cur_expert_num, self.expert_size])
            cur_gate_expert = paddle.multiply(x=expert_concat, y=cur_gate)
            cur_gate_expert = paddle.sum(x=cur_gate_expert, axis=1)
            outputs.append(cur_gate_expert)

        return outputs
