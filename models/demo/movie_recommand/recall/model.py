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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase
from recall_net import DNNLayer


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.is_distributed = True if envs.get_fleet_mode().upper(
        ) == "PSLIB" else False
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim")
        self.learning_rate = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate")
        self.hidden_layers = envs.get_global_env("hyper_parameters.fc_sizes")
        self.batch_size = envs.get_global_env("dygraph.batch_size")

    def net(self, input, is_infer=False):
        self.user_sparse_inputs = self._sparse_data_var[2:6]
        self.mov_sparse_inputs = self._sparse_data_var[6:9]
        self.label_input = self._sparse_data_var[-1]

        recall_model = DNNLayer(self.sparse_feature_number,
                                self.sparse_feature_dim, self.hidden_layers)

        predict = recall_model(self.batch_size, self.user_sparse_inputs,
                               self.mov_sparse_inputs, self.label_input)

        self.predict = predict

        if is_infer:
            self._infer_results["uid"] = self._sparse_data_var[2]
            self._infer_results["movieid"] = self._sparse_data_var[6]
            self._infer_results["label"] = self._sparse_data_var[-1]
            self._infer_results["predict"] = self.predict
            return

        cost = F.square_error_cost(
            self.predict, paddle.cast(
                x=self.label_input, dtype='float32'))
        avg_cost = paddle.mean(cost)
        self._cost = avg_cost
        self._metrics["LOSS"] = avg_cost
