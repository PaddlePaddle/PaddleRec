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


class ESCMLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim, num_field,
                 ctr_layer_sizes, cvr_layer_sizes, expert_num, expert_size,
                 tower_size, counterfact_mode, feature_size):
        super(ESCMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.num_field = num_field
        self.ctr_layer_sizes = ctr_layer_sizes
        self.cvr_layer_sizes = cvr_layer_sizes
        self.counterfact_mode = counterfact_mode
        self.expert_num = expert_num
        self.expert_size = expert_size
        self.tower_size = tower_size
        if counterfact_mode == "DR":
            self.gate_num = 3
        else:
            self.gate_num = 2
        self.feature_size = feature_size

        use_sparse = True
        if paddle.is_compiled_with_custom_device('npu'):
            use_sparse = False

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=use_sparse,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self._param_expert = []
        for i in range(0, self.expert_num):
            linear = self.add_sublayer(
                name='expert_' + str(i),
                sublayer=nn.Linear(
                    self.feature_size,
                    self.expert_size,
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

    def forward(self, inputs):
        emb = []
        # input feature data
        for data in inputs:
            feat_emb = self.embedding(data)
            # sum pooling
            feat_emb = paddle.sum(feat_emb, axis=1)
            emb.append(feat_emb)
        concat_emb = paddle.concat(x=emb, axis=1)
        expert_outputs = []
        for i in range(0, self.expert_num):
            linear_out = self._param_expert[i](concat_emb)
            expert_output = F.relu(linear_out)
            expert_outputs.append(expert_output)
        expert_concat = paddle.concat(x=expert_outputs, axis=1)
        expert_concat = paddle.reshape(
            expert_concat, [-1, self.expert_num, self.expert_size])

        output_layers = []
        for i in range(0, self.gate_num):
            cur_gate_linear = self._param_gate[i](concat_emb)
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
        out_list = []
        ctr_out = output_layers[0]
        cvr_out = output_layers[1]

        ctr_prop_one = paddle.slice(ctr_out, axes=[1], starts=[1], ends=[2])
        cvr_prop_one = paddle.slice(cvr_out, axes=[1], starts=[1], ends=[2])
        ctcvr_prop_one = paddle.multiply(x=ctr_prop_one, y=cvr_prop_one)
        ctcvr_prop = paddle.concat(
            x=[1 - ctcvr_prop_one, ctcvr_prop_one], axis=1)

        out_list = []
        out_list.append(ctr_out)
        out_list.append(ctr_prop_one)
        out_list.append(cvr_out)
        out_list.append(cvr_prop_one)
        out_list.append(ctcvr_prop)
        out_list.append(ctcvr_prop_one)
        if self.counterfact_mode == "DR":
            imp_out = output_layers[2]
            imp_prop_one = paddle.slice(
                imp_out, axes=[1], starts=[1], ends=[2])
            out_list.append(imp_prop_one)

        return out_list
