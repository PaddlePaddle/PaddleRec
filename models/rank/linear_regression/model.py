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

import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number", None)
        self.reg = envs.get_global_env("hyper_parameters.reg", 1e-4)

    def net(self, inputs, is_infer=False):
        init_value_ = 0.1
        is_distributed = True if envs.get_trainer() == "CtrTrainer" else False

        # ------------------------- network input --------------------------

        sparse_var = self._sparse_data_var
        self.label = self._dense_data_var[0]

        def embedding_layer(input):
            emb = fluid.embedding(
                input=input,
                is_sparse=True,
                is_distributed=is_distributed,
                size=[self.sparse_feature_number + 1, 1],
                padding_idx=0,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_),
                    regularizer=fluid.regularizer.L1DecayRegularizer(
                        self.reg)))
            reshape_emb = fluid.layers.reshape(emb, shape=[-1, 1])
            return reshape_emb

        sparse_embed_seq = list(map(embedding_layer, sparse_var))
        weight = fluid.layers.concat(sparse_embed_seq, axis=0)
        weight_sum = fluid.layers.reduce_sum(weight)
        b_linear = fluid.layers.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=fluid.initializer.ConstantInitializer(value=0))

        self.predict = fluid.layers.relu(weight_sum + b_linear)
        cost = fluid.layers.square_error_cost(
            input=self.predict, label=self.label)
        avg_cost = fluid.layers.reduce_sum(cost)

        self._cost = avg_cost

        self._metrics["COST"] = self._cost
        self._metrics["Predict"] = self.predict
        if is_infer:
            self._infer_results["Predict"] = self.predict
            self._infer_results["COST"] = self._cost
