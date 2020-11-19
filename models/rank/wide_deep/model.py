#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.hidden1_units = envs.get_global_env(
            "hyper_parameters.hidden1_units", 75)
        self.hidden2_units = envs.get_global_env(
            "hyper_parameters.hidden2_units", 50)
        self.hidden3_units = envs.get_global_env(
            "hyper_parameters.hidden3_units", 25)

    def wide_part(self, data):
        out = paddle.static.nn.fc(
            x=data,
            size=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.fluid.initializer.
                TruncatedNormalInitializer(
                    loc=0.0, scale=1.0 / math.sqrt(data.shape[1])),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4)),
            activation=None,
            name='wide')
        return out

    def fc(self, data, hidden_units, active, tag):
        output = paddle.static.nn.fc(
            x=data,
            size=hidden_units,
            weight_attr=paddle.ParamAttr(initializer=paddle.fluid.initializer.
                                         TruncatedNormalInitializer(
                                             loc=0.0,
                                             scale=1.0 /
                                             math.sqrt(data.shape[1]))),
            activation=active,
            name=tag)

        return output

    def deep_part(self, data, hidden1_units, hidden2_units, hidden3_units):
        l1 = self.fc(data, hidden1_units, 'relu', 'l1')
        l2 = self.fc(l1, hidden2_units, 'relu', 'l2')
        l3 = self.fc(l2, hidden3_units, 'relu', 'l3')

        return l3

    def net(self, inputs, is_infer=False):
        wide_input = self._dense_data_var[0]
        deep_input = self._dense_data_var[1]
        label = self._sparse_data_var[0]

        wide_output = self.wide_part(wide_input)
        deep_output = self.deep_part(deep_input, self.hidden1_units,
                                     self.hidden2_units, self.hidden3_units)

        wide_model = paddle.static.nn.fc(
            x=wide_output,
            size=1,
            weight_attr=paddle.ParamAttr(initializer=paddle.fluid.initializer.
                                         TruncatedNormalInitializer(
                                             loc=0.0, scale=1.0)),
            activation=None,
            name='w_wide')

        deep_model = paddle.static.nn.fc(
            x=deep_output,
            size=1,
            weight_attr=paddle.ParamAttr(initializer=paddle.fluid.initializer.
                                         TruncatedNormalInitializer(
                                             loc=0.0, scale=1.0)),
            activation=None,
            name='w_deep')

        prediction = paddle.add(x=wide_model, y=deep_model)
        pred = paddle.nn.functional.sigmoid(
            paddle.clip(
                prediction, min=-15.0, max=15.0), name="prediction")

        num_seqs = paddle.fluid.layers.create_tensor(dtype='int64')
        acc = paddle.metric.accuracy(
            input=pred,
            label=paddle.cast(
                x=label, dtype='int64'),
            total=num_seqs)
        auc_var, batch_auc, auc_states = paddle.fluid.layers.auc(
            input=pred, label=paddle.cast(
                x=label, dtype='int64'))

        self._metrics["AUC"] = auc_var
        self._metrics["BATCH_AUC"] = batch_auc
        self._metrics["ACC"] = acc
        if is_infer:
            self._infer_results["AUC"] = auc_var
            self._infer_results["ACC"] = acc

        cost = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(
            x=prediction, label=paddle.cast(
                label, dtype='float32'))
        avg_cost = paddle.mean(x=cost)
        self._cost = avg_cost
