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
import numpy as np


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def fc(self,tag, data, out_dim, active='prelu'):
        
        init_stddev = 1.0
        scales = 1.0  / np.sqrt(data.shape[1])
        
        p_attr = fluid.param_attr.ParamAttr(name='%s_weight' % tag,
                    initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=init_stddev * scales))
                    
        b_attr = fluid.ParamAttr(name='%s_bias' % tag, initializer=fluid.initializer.Constant(0.1))
        
        out = fluid.layers.fc(input=data,
                            size=out_dim,
                            act=active,
                            param_attr=p_attr, 
                            bias_attr =b_attr,
                            name=tag)
        return out
        
    def input_data(self):
        sparse_input_ids = [
            fluid.data(name="field_" + str(i), shape=[-1, 1], dtype="int64", lod_level=1) for i in range(0,23)
        ]
        label_ctr = fluid.data(name="ctr", shape=[-1, 1], dtype="int64")
        label_cvr = fluid.data(name="cvr", shape=[-1, 1], dtype="int64")
        inputs = sparse_input_ids + [label_ctr] + [label_cvr]
        self._data_var.extend(inputs)
        
        return inputs
    
    def net(self, inputs):
        
        vocab_size = envs.get_global_env("hyper_parameters.vocab_size", None, self._namespace)
        embed_size = envs.get_global_env("hyper_parameters.embed_size", None, self._namespace)
        emb = []
        for data in inputs[0:-2]:
            feat_emb = fluid.embedding(input=data,
                                size=[vocab_size, embed_size],
                                param_attr=fluid.ParamAttr(name='dis_emb',
                                                            learning_rate=5,
                                                            initializer=fluid.initializer.Xavier(fan_in=embed_size,fan_out=embed_size)
                                                            ),
                                is_sparse=True)
            field_emb = fluid.layers.sequence_pool(input=feat_emb,pool_type='sum')
            emb.append(field_emb)
        concat_emb = fluid.layers.concat(emb, axis=1)
        
        # ctr
        active = 'relu'
        ctr_fc1 = self.fc('ctr_fc1', concat_emb, 200, active)
        ctr_fc2 = self.fc('ctr_fc2', ctr_fc1, 80, active)
        ctr_out = self.fc('ctr_out', ctr_fc2, 2, 'softmax')
        
        # cvr
        cvr_fc1 = self.fc('cvr_fc1', concat_emb, 200, active)
        cvr_fc2 = self.fc('cvr_fc2', cvr_fc1, 80, active)
        cvr_out = self.fc('cvr_out', cvr_fc2, 2,'softmax')
    
        ctr_clk = inputs[-2]
        ctcvr_buy = inputs[-1]
        
        ctr_prop_one = fluid.layers.slice(ctr_out, axes=[1], starts=[1], ends=[2])
        cvr_prop_one = fluid.layers.slice(cvr_out, axes=[1], starts=[1], ends=[2])
        
        ctcvr_prop_one = fluid.layers.elementwise_mul(ctr_prop_one, cvr_prop_one)
        ctcvr_prop = fluid.layers.concat(input=[1-ctcvr_prop_one,ctcvr_prop_one], axis = 1)
    
        loss_ctr = fluid.layers.cross_entropy(input=ctr_out, label=ctr_clk)
        loss_ctcvr = fluid.layers.cross_entropy(input=ctcvr_prop, label=ctcvr_buy)
        cost = loss_ctr + loss_ctcvr
        avg_cost = fluid.layers.mean(cost)

        auc_ctr, batch_auc_ctr, auc_states_ctr = fluid.layers.auc(input=ctr_out, label=ctr_clk)
        auc_ctcvr, batch_auc_ctcvr, auc_states_ctcvr = fluid.layers.auc(input=ctcvr_prop, label=ctcvr_buy)
    
        self._cost = avg_cost
        self._metrics["AUC_ctr"] = auc_ctr
        self._metrics["BATCH_AUC_ctr"] = batch_auc_ctr
        self._metrics["AUC_ctcvr"] = auc_ctcvr
        self._metrics["BATCH_AUC_ctcvr"] = batch_auc_ctcvr


    def train_net(self):
        input_data = self.input_data()
        self.net(input_data)


    def infer_net(self):
        pass
