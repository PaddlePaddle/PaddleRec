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
from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase
from dssm_net import DSSMLayer


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.trigram_d = envs.get_global_env("hyper_parameters.trigram_d")
        self.neg_num = envs.get_global_env("hyper_parameters.neg_num")
        self.hidden_layers = envs.get_global_env("hyper_parameters.fc_sizes")
        self.hidden_acts = envs.get_global_env("hyper_parameters.fc_acts")
        self.learning_rate = envs.get_global_env(
            "hyper_parameters.learning_rate")
        self.slice_end = envs.get_global_env("hyper_parameters.slice_end")

    def input_data(self, is_infer=False, **kwargs):
        query = paddle.static.data(
            name="query",
            shape=[-1, self.trigram_d],
            dtype='float32',
            lod_level=0)
        doc_pos = paddle.static.data(
            name="doc_pos",
            shape=[-1, self.trigram_d],
            dtype='float32',
            lod_level=0)

        if is_infer:
            return [query, doc_pos]

        doc_negs = [
            paddle.static.data(
                name="doc_neg_" + str(i),
                shape=[-1, self.trigram_d],
                dtype="float32",
                lod_level=0) for i in range(self.neg_num)
        ]
        return [query, doc_pos] + doc_negs

    def net(self, inputs, is_infer=False):
        dssm_model = DSSMLayer(self.trigram_d, self.neg_num, self.slice_end,
                               self.hidden_layers, self.hidden_acts)
        R_Q_D_p, hit_prob = dssm_model(inputs, is_infer)

        if is_infer:
            self._infer_results["query_doc_sim"] = R_Q_D_p
            return

        loss = -paddle.sum(paddle.log(hit_prob))
        avg_cost = paddle.mean(x=loss)
        self._cost = avg_cost
        self._metrics["LOSS"] = avg_cost
