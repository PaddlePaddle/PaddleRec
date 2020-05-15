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

import math
import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import Model as ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def MMOE(self, is_infer=False):

        feature_size = envs.get_global_env("hyper_parameters.feature_size", None, self._namespace)
        expert_num = envs.get_global_env("hyper_parameters.expert_num", None, self._namespace)
        gate_num = envs.get_global_env("hyper_parameters.gate_num", None, self._namespace)
        expert_size = envs.get_global_env("hyper_parameters.expert_size", None, self._namespace)
        tower_size = envs.get_global_env("hyper_parameters.tower_size", None, self._namespace)

        input_data = fluid.data(name="input", shape=[-1, feature_size], dtype="float32")
        label_income = fluid.data(name="label_income", shape=[-1, 2], dtype="float32", lod_level=0)
        label_marital = fluid.data(name="label_marital", shape=[-1, 2], dtype="float32", lod_level=0)
        if is_infer:
            self._infer_data_var = [input_data, label_income, label_marital]
            self._infer_data_loader = fluid.io.DataLoader.from_generator(
                    feed_list=self._infer_data_var, capacity=64, use_double_buffer=False, iterable=False)
        
        self._data_var.extend([input_data, label_income, label_marital])
        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper
        expert_outputs = []
        for i in range(0, expert_num):
            expert_output = fluid.layers.fc(input=input_data,
                                           size=expert_size,
                                           act='relu',
                                           bias_attr=fluid.ParamAttr(learning_rate=1.0),
                                           name='expert_' + str(i))
            expert_outputs.append(expert_output)
        expert_concat = fluid.layers.concat(expert_outputs, axis=1)
        expert_concat = fluid.layers.reshape(expert_concat,[-1, expert_num, expert_size])
        
        
        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper
        output_layers = []
        for i in range(0, gate_num):
            cur_gate = fluid.layers.fc(input=input_data,
                                       size=expert_num,
                                       act='softmax',
                                       bias_attr=fluid.ParamAttr(learning_rate=1.0),
                                       name='gate_' + str(i))
            # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
            cur_gate_expert = fluid.layers.elementwise_mul(expert_concat, cur_gate, axis=0)  
            cur_gate_expert = fluid.layers.reduce_sum(cur_gate_expert, dim=1)
            # Build tower layer
            cur_tower =  fluid.layers.fc(input=cur_gate_expert,
                                      size=tower_size,
                                      act='relu',
                                      name='task_layer_' + str(i))  
            out =  fluid.layers.fc(input=cur_tower,
                                   size=2,
                                   act='softmax',
                                   name='out_' + str(i))
                
            output_layers.append(out)

        pred_income = fluid.layers.clip(output_layers[0], min=1e-15, max=1.0 - 1e-15)
        pred_marital = fluid.layers.clip(output_layers[1], min=1e-15, max=1.0 - 1e-15)

        
        label_income_1 = fluid.layers.slice(label_income, axes=[1], starts=[1], ends=[2])
        label_marital_1 = fluid.layers.slice(label_marital, axes=[1], starts=[1], ends=[2])
        
        auc_income, batch_auc_1, auc_states_1  = fluid.layers.auc(input=pred_income, label=fluid.layers.cast(x=label_income_1, dtype='int64'))
        auc_marital, batch_auc_2, auc_states_2 = fluid.layers.auc(input=pred_marital, label=fluid.layers.cast(x=label_marital_1, dtype='int64'))
        if is_infer:
            self._infer_results["AUC_income"] = auc_income
            self._infer_results["AUC_marital"] = auc_marital
            return

        cost_income = fluid.layers.cross_entropy(input=pred_income, label=label_income,soft_label = True)
        cost_marital = fluid.layers.cross_entropy(input=pred_marital, label=label_marital,soft_label = True)
        
        avg_cost_income = fluid.layers.mean(x=cost_income)
        avg_cost_marital = fluid.layers.mean(x=cost_marital)
        
        cost =  avg_cost_income + avg_cost_marital
    
        self._cost = cost
        self._metrics["AUC_income"] = auc_income
        self._metrics["BATCH_AUC_income"] = batch_auc_1
        self._metrics["AUC_marital"] = auc_marital
        self._metrics["BATCH_AUC_marital"] = batch_auc_2


    def train_net(self):
        self.MMOE()


    def infer_net(self):
        self.MMOE(is_infer=True)
