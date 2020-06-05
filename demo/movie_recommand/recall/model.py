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

    def _init_hyper_parameters(self):
        self.is_distributed = True if envs.get_trainer(
        ) == "CtrTrainer" else False
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim")
        self.learning_rate = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate")
        self.hidden_layers = envs.get_global_env("hyper_parameters.fc_sizes")

    def net(self, input, is_infer=False):
        self.user_sparse_inputs = self._sparse_data_var[2:6]
        self.mov_sparse_inputs = self._sparse_data_var[6:9]

        self.label_input = self._sparse_data_var[-1]

        def fc(input):
            fcs = [input]
            for size in self.hidden_layers:
                output = fluid.layers.fc(
                    input=fcs[-1],
                    size=size,
                    act='relu',
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Normal(
                            scale=1.0 / math.sqrt(fcs[-1].shape[1]))))
                fcs.append(output)
            return fcs[-1]

        def embedding_layer(input):
            emb = fluid.layers.embedding(
                input=input,
                is_sparse=True,
                is_distributed=self.is_distributed,
                size=[self.sparse_feature_number, self.sparse_feature_dim],
                param_attr=fluid.ParamAttr(
                    name="emb", initializer=fluid.initializer.Uniform()), )
            emb_sum = fluid.layers.sequence_pool(input=emb, pool_type='sum')
            return emb_sum

        user_sparse_embed_seq = list(
            map(embedding_layer, self.user_sparse_inputs))
        mov_sparse_embed_seq = list(
            map(embedding_layer, self.mov_sparse_inputs))
        concated_user = fluid.layers.concat(user_sparse_embed_seq, axis=1)
        concated_mov = fluid.layers.concat(mov_sparse_embed_seq, axis=1)

        usr_combined_features = fc(concated_user)
        mov_combined_features = fc(concated_mov)

        sim = fluid.layers.cos_sim(
            X=usr_combined_features, Y=mov_combined_features)
        predict = fluid.layers.scale(sim, scale=5)
        self.predict = predict

        if is_infer:
            self._infer_results["uid"] = self._sparse_data_var[2]
            self._infer_results["movieid"] = self._sparse_data_var[6]
            self._infer_results["label"] = self._sparse_data_var[-1]
            self._infer_results["predict"] = self.predict
            return

        cost = fluid.layers.square_error_cost(
            self.predict,
            fluid.layers.cast(
                x=self.label_input, dtype='float32'))
        avg_cost = fluid.layers.reduce_mean(cost)
        self._cost = avg_cost
        self._metrics["LOSS"] = avg_cost

    def optimizer(self):
        optimizer = fluid.optimizer.Adam(self.learning_rate, lazy_mode=True)
        return optimizer
