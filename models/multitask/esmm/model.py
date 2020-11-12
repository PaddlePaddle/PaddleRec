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

import numpy as np
import paddle

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.vocab_size = envs.get_global_env("hyper_parameters.vocab_size")
        self.embed_size = envs.get_global_env("hyper_parameters.embed_size")

    def input_data(self, is_infer=False, **kwargs):
        sparse_input_ids = [
            paddle.fluid.data(
                name="field_" + str(i),
                shape=[-1, 1],
                dtype="int64",
                lod_level=1) for i in range(0, 23)
        ]
        label_ctr = paddle.fluid.data(
            name="ctr", shape=[-1, 1], dtype="float32")
        label_cvr = paddle.fluid.data(
            name="cvr", shape=[-1, 1], dtype="float32")
        inputs = sparse_input_ids + [label_ctr] + [label_cvr]
        if is_infer:
            return inputs
        else:
            return inputs

    def net(self, inputs, is_infer=False):

        emb = []
        # input feature data
        for data in inputs[0:-2]:
            feat_emb = paddle.static.nn.embedding(
                input=data,
                size=[self.vocab_size, self.embed_size],
                param_attr=paddle.ParamAttr(
                    name='dis_emb',
                    learning_rate=5,
                    initializer=paddle.fluid.initializer.Xavier(
                        fan_in=self.embed_size, fan_out=self.embed_size)),
                is_sparse=True)
            field_emb = paddle.fluid.layers.sequence_pool(
                input=feat_emb, pool_type='sum')
            emb.append(field_emb)
        concat_emb = paddle.concat(x=emb, axis=1)

        # ctr
        active = 'relu'
        ctr_fc1 = self._fc('ctr_fc1', concat_emb, 200, active)
        ctr_fc2 = self._fc('ctr_fc2', ctr_fc1, 80, active)
        ctr_out = self._fc('ctr_out', ctr_fc2, 2, 'softmax')

        # cvr
        cvr_fc1 = self._fc('cvr_fc1', concat_emb, 200, active)
        cvr_fc2 = self._fc('cvr_fc2', cvr_fc1, 80, active)
        cvr_out = self._fc('cvr_out', cvr_fc2, 2, 'softmax')

        ctr_clk = inputs[-2]
        ctr_clk_one = paddle.slice(ctr_clk, axes=[1], starts=[0], ends=[1])
        ctcvr_buy = inputs[-1]
        ctcvr_buy_one = paddle.slice(ctcvr_buy, axes=[1], starts=[0], ends=[1])

        ctr_prop_one = paddle.slice(ctr_out, axes=[1], starts=[0], ends=[1])
        cvr_prop_one = paddle.slice(cvr_out, axes=[1], starts=[0], ends=[1])

        ctcvr_prop_one = paddle.multiply(x=ctr_prop_one, y=cvr_prop_one)
        ctcvr_prop = paddle.concat(
            x=[1 - ctcvr_prop_one, ctcvr_prop_one], axis=1)

        auc_ctr, batch_auc_ctr, auc_states_ctr = paddle.fluid.layers.auc(
            input=ctr_out, label=paddle.cast(
                x=ctr_clk, dtype='int64'))
        auc_ctcvr, batch_auc_ctcvr, auc_states_ctcvr = paddle.fluid.layers.auc(
            input=ctcvr_prop, label=paddle.cast(
                x=ctcvr_buy, dtype='int64'))

        if is_infer:
            self._infer_results["AUC_ctr"] = auc_ctr
            self._infer_results["AUC_ctcvr"] = auc_ctcvr
            return

        loss_ctr = paddle.nn.functional.log_loss(
            input=ctr_prop_one, label=ctr_clk_one)
        loss_ctcvr = paddle.nn.functional.log_loss(
            input=ctcvr_prop_one, label=ctcvr_buy_one)
        cost = loss_ctr + loss_ctcvr
        avg_cost = paddle.mean(x=cost)

        self._cost = avg_cost
        self._metrics["AUC_ctr"] = auc_ctr
        self._metrics["BATCH_AUC_ctr"] = batch_auc_ctr
        self._metrics["AUC_ctcvr"] = auc_ctcvr
        self._metrics["BATCH_AUC_ctcvr"] = batch_auc_ctcvr

    def _fc(self, tag, data, out_dim, active='prelu'):

        init_stddev = 1.0
        scales = 1.0 / np.sqrt(data.shape[1])

        p_attr = paddle.ParamAttr(
            name='%s_weight' % tag,
            initializer=paddle.fluid.initializer.NormalInitializer(
                loc=0.0, scale=init_stddev * scales))

        b_attr = paddle.ParamAttr(
            name='%s_bias' % tag,
            initializer=paddle.nn.initializer.Constant(value=0.1))

        out = paddle.static.nn.fc(x=data,
                                  size=out_dim,
                                  activation=active,
                                  weight_attr=p_attr,
                                  bias_attr=b_attr,
                                  name=tag)
        return out
