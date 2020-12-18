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
import paddle.nn.functional as F

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase
from wide_deep_net import WideDeepLayer


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.hidden1_units = envs.get_global_env(
            "hyper_parameters.hidden1_units")
        self.hidden2_units = envs.get_global_env(
            "hyper_parameters.hidden2_units")
        self.hidden3_units = envs.get_global_env(
            "hyper_parameters.hidden3_units")
        self.wide_input_dim = envs.get_global_env(
            "hyper_parameters.wide_input_dim")
        self.deep_input_dim = envs.get_global_env(
            "hyper_parameters.deep_input_dim")
        self.layer_sizes = [
            self.hidden1_units, self.hidden2_units, self.hidden3_units
        ]

    def net(self, inputs, is_infer=False):
        wide_input = self._dense_data_var[0]
        deep_input = self._dense_data_var[1]
        label = self._sparse_data_var[0]

        wide_deep_model = WideDeepLayer(self.wide_input_dim,
                                        self.deep_input_dim, self.layer_sizes)
        prediction = wide_deep_model.forward(wide_input, deep_input)

        pred = F.sigmoid(prediction)

        acc = paddle.metric.accuracy(
            input=pred, label=paddle.cast(
                x=label, dtype='int64'))
        auc_var, batch_auc, auc_states = paddle.fluid.layers.auc(
            input=pred, label=paddle.cast(
                x=label, dtype='int64'))

        self._metrics["AUC"] = auc_var
        self._metrics["BATCH_AUC"] = batch_auc
        self._metrics["ACC"] = acc
        if is_infer:
            self._infer_results["AUC"] = auc_var
            self._infer_results["ACC"] = acc

        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                label, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        self._cost = avg_cost
