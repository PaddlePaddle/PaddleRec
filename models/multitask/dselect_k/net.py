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
import math
import numpy as np


class ExpertLayer(nn.Layer):
    def __init__(self, output_size=8, use_bn=True):
        super(ExpertLayer, self).__init__()
        self.use_bn = use_bn

        # out: 32 * 32 (36 - 5 + 1 = 32)
        self.conv1 = paddle.nn.Conv2D(
            in_channels=1,
            out_channels=10,
            kernel_size=5,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal()))
        self.bn1 = paddle.nn.BatchNorm2D(num_features=10)
        # out: 16 * 16 (32/2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2)
        # out: 12 * 12 (16 - 5 + 1)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=10,
            out_channels=20,
            kernel_size=5,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal()))
        self.bn2 = paddle.nn.BatchNorm2D(num_features=20)
        # out: 6 * 6 (12/2)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2)
        self.linear1 = paddle.nn.Linear(
            in_features=20 * 6 * 6,
            out_features=50,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    std=1.0 / math.sqrt(720))))
        self.linear2 = paddle.nn.Linear(
            in_features=50,
            out_features=50,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    std=1.0 / math.sqrt(50))))
        self.linear3 = paddle.nn.Linear(
            in_features=50,
            out_features=output_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    std=1.0 / math.sqrt(50))))

    def forward(self, x):
        x = self.conv1(x)

        if self.use_bn:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)

        if self.use_bn:
            x = self.bn2(x)

        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        # x = self.linear2(x)
        # x = F.relu(x)
        x = self.linear3(x)
        return x


class TowerLayer(nn.Layer):
    def __init__(self, input_size, output_size=10):
        super(TowerLayer, self).__init__()
        self.linear1 = paddle.nn.Linear(
            in_features=input_size,
            out_features=50,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    std=1.0 / math.sqrt(input_size))))
        self.linear2 = paddle.nn.Linear(
            in_features=50,
            out_features=50,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    std=1.0 / math.sqrt(input_size))))
        self.linear3 = paddle.nn.Linear(
            in_features=50,
            out_features=output_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    std=1.0 / math.sqrt(input_size))))

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


class SmoothStep(nn.Layer):
    def __init__(self, gamma=1.0):
        super(SmoothStep, self).__init__()
        self._lower_bound = -gamma / 2.0
        self._upper_bound = gamma / 2.0
        self._a3 = -2 / (gamma**3)
        self._a1 = 3 / (2 * gamma)
        self._a0 = 0.5

    def forward(self, inputs):
        return paddle.where(
            condition=inputs <= self._lower_bound,
            x=paddle.zeros_like(inputs),
            y=paddle.where(
                condition=inputs >= self._upper_bound,
                x=paddle.ones_like(inputs),
                y=self._a3 * (inputs**3) + self._a1 * inputs + self._a0))


class DSelectkGate(nn.Layer):
    def __init__(self,
                 num_nonzeros,
                 expert_num,
                 input_shape=None,
                 gamma=1.0,
                 entropy_reg=None,
                 z_initializer=None,
                 w_initializer=None):
        super(DSelectkGate, self).__init__()
        self._num_nonzeros = num_nonzeros
        self._smooth_step = SmoothStep(gamma)
        self._entropy_reg = entropy_reg
        self._z_initializer = z_initializer or paddle.nn.initializer.Uniform(
            -gamma / 100.0, gamma / 100.0)
        self._w_initializer = w_initializer or paddle.nn.initializer.Uniform()
        self._expert_num = expert_num
        self._num_binary = math.ceil(math.log2(expert_num))
        self._power_of_2 = (expert_num == 2**self._num_binary)
        if input_shape is None:
            z_logits = self.create_parameter(
                shape=[self._num_nonzeros, 1, self._num_binary],
                attr=paddle.ParamAttr(initializer=self._z_initializer))
            self._z_logits = z_logits
            self.add_parameter("z_logits", z_logits)

            w_logits = self.create_parameter(
                shape=[self._num_nonzeros, 1],
                attr=paddle.ParamAttr(initializer=self._w_initializer))
            self._w_logits = w_logits
            self.add_parameter("w_logits", w_logits)
        else:
            self._z_logits = paddle.nn.Linear(
                in_features=input_shape,
                out_features=self._num_nonzeros * self._num_binary,
                weight_attr=paddle.ParamAttr(initializer=self._z_initializer),
                bias_attr=paddle.ParamAttr(initializer=self._z_initializer))
            self._w_logits = paddle.nn.Linear(
                in_features=input_shape,
                out_features=self._num_nonzeros,
                weight_attr=paddle.ParamAttr(initializer=self._w_initializer),
                bias_attr=paddle.ParamAttr(initializer=self._w_initializer))

        binary_matrix = np.array([
            list(np.binary_repr(
                val, width=self._num_binary)) for val in range(expert_num)
        ]).astype(bool)

        if not paddle.in_dynamic_mode():
            # 兼容静态图
            self._binary_codes = paddle.cast(
                paddle.ones(shape=[1, expert_num, self._num_binary]),
                dtype=bool)
        else:
            self._binary_codes = paddle.unsqueeze(
                paddle.to_tensor(
                    binary_matrix, dtype=bool), axis=0)

    def forward(self, inputs, training=False):
        if isinstance(inputs, tuple):
            experts, routing_inputs = inputs
        else:
            experts, routing_inputs = inputs, None

        if routing_inputs is None:
            # static gating
            expert_weights, selector_outputs = self._compute_expert_weights()
            output = paddle.add_n(inputs=[
                expert_weights[i] * experts[i] for i in range(len(experts))
            ])
        else:
            # per-example gating
            expert_weights, selector_outputs = self._compute_example_conditioned_expert_weights(
                routing_inputs)
            output = paddle.add_n(inputs=[
                paddle.reshape(expert_weights[:, i], [-1, 1]) * experts[i]
                for i in range(len(experts))
            ])

        return output

    def _compute_expert_weights(self):
        """Computes the weight vector for the experts.
        Args: None.
        Returns:
          A tuple: (expert_weights, selector_outputs).
            expert_weights is the final weight vector of the experts.
            selector_outputs is a (num_nonzero, num_experts)-matrix whose i-th row
            represents the outputs of the i-th single-expert selector.
        """
        # Shape = (num_nonzero, 1, num_binary)
        smooth_step_activations = self._smooth_step(self._z_logits)

        # Shape = (num_nonzero, num_experts)
        selector_outputs = paddle.prod(
            paddle.where(self._binary_codes, smooth_step_activations,
                         1 - smooth_step_activations),
            axis=2)

        # Weights for the single-expert selectors: shape = (num_nonzero, 1)
        selector_weights = F.softmax(self._w_logits, axis=0)
        expert_weights = paddle.sum(selector_weights * selector_outputs,
                                    axis=0)

        return expert_weights, selector_outputs

    def _compute_example_conditioned_expert_weights(self, routing_inputs):
        """Computes the example-conditioned weights for the experts.
        Args:
            routing_inputs: a tensor of shape=(batch_size, num_features) containing
            the input examples.
        Returns:
            A tuple: (expert_weights, selector_outputs).
            expert_weights is a tensor with shape=(batch_size, num_experts),
            containing the expert weights for each example in routing_inputs.
            selector_outputs is a tensor with
            shape=(batch_size, num_nonzero, num_experts), which contains the outputs
            of the single-expert selectors for all the examples in routing_inputs.
        """
        sample_logits = paddle.reshape(
            self._z_logits(routing_inputs),
            [-1, self._num_nonzeros, 1, self._num_binary])
        smooth_step_activations = self._smooth_step(sample_logits)

        # Shape = (batch_size, num_nonzeros, num_experts).
        selector_outputs = paddle.prod(
            paddle.where(
                paddle.unsqueeze(self._binary_codes, 0),
                smooth_step_activations, 1 - smooth_step_activations),
            axis=3)

        # Weights for the single-expert selectors
        # Shape = (batch_size, num_nonzeros, 1)
        selector_weights = paddle.unsqueeze(self._w_logits(routing_inputs), 2)
        selector_weights = F.softmax(selector_weights, axis=1)

        # Sum over the signle-expert selectors. Shape = (batch_size, num_experts).
        expert_weights = paddle.sum(selector_weights * selector_outputs,
                                    axis=1)

        return expert_weights, selector_outputs


class MMoELayer(nn.Layer):
    def __init__(self,
                 feature_size,
                 expert_num,
                 expert_size,
                 tower_size,
                 gate_num,
                 topk=2):
        super(MMoELayer, self).__init__()

        self.expert_num = expert_num
        self.expert_size = expert_size
        self.tower_size = tower_size
        self.gate_num = gate_num
        self.num_nonzeros = topk

        self._dselect_k = DSelectkGate(
            expert_num=expert_num, num_nonzeros=topk)

        self._param_expert = []
        for i in range(0, self.expert_num):
            cnn = self.add_sublayer(
                name="expert_" + str(i),
                sublayer=ExpertLayer(output_size=expert_size))
            self._param_expert.append(cnn)

        self._param_gate = []
        self._param_tower_out = []
        for i in range(0, self.gate_num):
            cnn = self.add_sublayer(
                name="gate_" + str(i),
                sublayer=ExpertLayer(output_size=expert_num))
            self._param_gate.append(cnn)

            tower = self.add_sublayer(
                name='tower_out_' + str(i),
                sublayer=TowerLayer(
                    input_size=50, output_size=10))
            self._param_tower_out.append(tower)

    def forward(self, input_data):
        expert_outputs = []
        for i in range(0, self.expert_num):
            linear_out = self._param_expert[i](input_data)
            expert_output = F.relu(linear_out)
            expert_outputs.append(expert_output)

        # # MMoE
        # # (512, expert_size * 8)
        # expert_concat = paddle.concat(x=expert_outputs, axis=1)
        # expert_concat = paddle.reshape(
        #     expert_concat, [-1, self.expert_num, self.expert_size])
        #
        # output_layers = []
        # for i in range(0, self.gate_num):
        #     cur_gate_linear = self._param_gate[i](input_data)
        #     cur_gate = F.softmax(cur_gate_linear)
        #     cur_gate = paddle.reshape(cur_gate, [-1, self.expert_num, 1])
        #     cur_gate_expert = paddle.multiply(x=expert_concat, y=cur_gate)
        #     cur_gate_expert = paddle.sum(x=cur_gate_expert, axis=1)
        #     out = self._param_tower_out[i](cur_gate_expert)
        #     out = F.softmax(out)
        #     out = paddle.clip(out, min=1e-15, max=1.0 - 1e-15)
        #     output_layers.append(out)

        # DSelect-K
        gate_output = self._dselect_k(expert_outputs)
        output_layers = []
        for i in range(0, self.gate_num):
            out = self._param_tower_out[i](gate_output)
            out = F.softmax(out)
            out = paddle.clip(out, min=1e-15, max=1.0 - 1e-15)
            output_layers.append(out)

        return output_layers
