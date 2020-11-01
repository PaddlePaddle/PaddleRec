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
import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.dense_feature_dim = envs.get_global_env(
            "hyper_parameters.dense_feature_dim")
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim")
        self.learning_rate = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate")

    def input_data(self, is_infer=False, **kwargs):
        dense_input = paddle.data(
            name="dense_input",
            shape=[self.dense_feature_dim],
            dtype="float32")

        sparse_input_ids = [
            paddle.data(
                name="C" + str(i), shape=[1], lod_level=1, dtype="int64")
            for i in range(1, 27)
        ]

        label = paddle.data(name="label", shape=[1], dtype="float32")

        inputs = [dense_input] + sparse_input_ids + [label]
        return inputs

    def net(self, input, is_infer=False):
        self.dense_input = input[0]
        self.sparse_input = input[1:-1]
        self.label_input = self.input[-1]

        def embedding_layer(input):
            emb = paddle.static.nn.embedding(
                input=input,
                is_sparse=True,
                is_distributed=self.is_distributed,
                size=[self.sparse_feature_number, self.sparse_feature_dim],
                param_attr=paddle.ParamAttr(
                    name="SparseFeatFactors",
                    initializer=fluid.initializer.Uniform()))
            emb_sum = fluid.layers.sequence_pool(input=emb, pool_type='sum')
            return emb_sum

        sparse_embed_seq = list(map(embedding_layer, self.sparse_inputs))
        concated = paddle.concat(sparse_embed_seq + [self.dense_input], axis=1)

        fcs = [concated]
        hidden_layers = envs.get_global_env("hyper_parameters.fc_sizes")

        for size in hidden_layers:
            output = paddle.static.nn.fc(
                input=fcs[-1],
                size=size,
                act='relu',
                param_attr=paddle.ParamAttr(
                    initializer=fluid.initializer.Normal(
                        scale=1.0 / math.sqrt(fcs[-1].shape[1]))))
            fcs.append(output)

        predict = paddle.static.nn.fc(
            input=fcs[-1],
            size=2,
            act="softmax",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(fcs[-1].shape[1]))))

        self.predict = predict

        auc, batch_auc, _ = fluid.layers.auc(input=self.predict,
                                             label=self.label_input,
                                             num_thresholds=2**12,
                                             slide_steps=20)
        if is_infer:
            self._infer_results["AUC"] = auc
            self._infer_results["BATCH_AUC"] = batch_auc
            return

        self._metrics["AUC"] = auc
        self._metrics["BATCH_AUC"] = batch_auc
        cost = paddle.nn.functional.cross_entropy(
            input=self.predict, label=self.label_input)
        avg_cost = fluid.layers.reduce_mean(cost)
        self._cost = avg_cost

    def optimizer(self):
        optimizer = paddle.optimizer.Adam(self.learning_rate, lazy_mode=True)
        return optimizer

    def infer_net(self):
        pass
