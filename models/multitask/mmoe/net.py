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


class MMoELayer(nn.Layer):
    def __init__(self, feature_size, expert_num, expert_size, tower_size,
                 gate_num):
        super(MMoELayer, self).__init__()

        self.expert_num = expert_num
        self.expert_size = expert_size
        self.tower_size = tower_size
        self.gate_num = gate_num

        self._param_expert = []
        for i in range(0, self.expert_num):
            linear = self.add_sublayer(
                name='expert_' + str(i),
                sublayer=nn.Linear(
                    feature_size,
                    expert_size,
                    #initialize the weight randly
                    weight_attr=nn.initializer.XavierUniform(),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    #bias_attr=paddle.ParamAttr(learning_rate=1.0),
                    name='expert_' + str(i)))
            self._param_expert.append(linear)

        self._param_gate = []
        self._param_tower = []
        self._param_tower_out = []
        for i in range(0, self.gate_num):
            linear = self.add_sublayer(
                name='gate_' + str(i),
                sublayer=nn.Linear(
                    feature_size,
                    expert_num,
                    #initialize the weight randly
                    weight_attr=nn.initializer.XavierUniform(),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    #bias_attr=paddle.ParamAttr(learning_rate=1.0),
                    name='gate_' + str(i)))
            self._param_gate.append(linear)

            linear = self.add_sublayer(
                name='tower_' + str(i),
                sublayer=nn.Linear(
                    expert_size,
                    tower_size,
                    #initialize the weight randly
                    weight_attr=nn.initializer.XavierUniform(),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    #bias_attr=paddle.ParamAttr(learning_rate=1.0),
                    name='tower_' + str(i)))
            self._param_tower.append(linear)

            linear = self.add_sublayer(
                name='tower_out_' + str(i),
                sublayer=nn.Linear(
                    tower_size,
                    2,
                    #initialize the weight randly
                    weight_attr=nn.initializer.XavierUniform(),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    name='tower_out_' + str(i)))
            self._param_tower_out.append(linear)

    def forward(self, input_data):
        expert_outputs = []
        for i in range(0, self.expert_num):
            linear_out = self._param_expert[i](input_data)
            expert_output = F.relu(linear_out)
            expert_outputs.append(expert_output)
        expert_concat = paddle.concat(x=expert_outputs, axis=1)
        expert_concat = paddle.reshape(
            expert_concat, [-1, self.expert_num, self.expert_size])

        output_layers = []
        for i in range(0, self.gate_num):
            cur_gate_linear = self._param_gate[i](input_data)
            cur_gate = F.softmax(cur_gate_linear)
            cur_gate = paddle.reshape(cur_gate, [-1, self.expert_num, 1])
            cur_gate_expert = paddle.multiply(x=expert_concat, y=cur_gate)
            cur_gate_expert = paddle.sum(x=cur_gate_expert, axis=1)
            cur_tower = self._param_tower[i](cur_gate_expert)
            cur_tower = F.relu(cur_tower)
            out = self._param_tower_out[i](cur_tower)
            out = F.softmax(out)
            out = paddle.clip(out, min=1e-15, max=1.0 - 1e-15)
            output_layers.append(out)

        return output_layers
