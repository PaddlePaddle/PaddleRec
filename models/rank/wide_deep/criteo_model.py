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
        self.hidden_size = envs.get_global_env(
            "hyper_parameters.hidden_size", [1024] * 7)
        self.dense_feature_dim = envs.get_global_env(
            "hyper_parameters.dense_feature_dim", 13)
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number", 3200000)
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim", 16)
        self.wide_feature_number = envs.get_global_env(
            "hyper_parameters.wide_feature_number", 3200000)
        self.learning_rate = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate", 1e-4)

    def input_data(self, is_infer=False, **kwargs):
        dense_input = paddle.static.data(
            name="dense_input",
            shape=[-1, self.dense_feature_dim],
            dtype="float32")

        sparse_input_ids = [
            paddle.static.data(
                name="C" + str(i), shape=[-1, 1], dtype="int64")
            for i in range(1, 27)
        ]

        wide_input_idx = [
            paddle.static.data(name="C_{}_{}".format(
                i*2 + 1, i*2 + 2), shape=[-1, 1], dtype="int64")
            for i in range(2)
        ]

        label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")

        inputs = [dense_input] + sparse_input_ids + wide_input_idx + [label]
        return inputs

    def net(self, input, is_infer=False):
        self.dense_input = input[0]
        self.sparse_input = input[1:27]
        self.wide_input = input[27:-1]
        self.label_input = input[-1]

        def sparse_embedding_layer(input):
            emb = paddle.static.nn.embedding(
                input=input,
                is_sparse=True,
                size=[self.sparse_feature_number, self.sparse_feature_dim],
                param_attr=paddle.ParamAttr(
                    name="sparse_embedding",
                    initializer=fluid.initializer.Uniform()))
            emb = paddle.reshape(emb, [-1, self.sparse_feature_dim])
            return emb

        def wide_embedding_layer(input):
            emb = paddle.static.nn.embedding(
                input=input,
                is_sparse=True,
                size=[self.wide_feature_number, 1],
                param_attr=paddle.ParamAttr(
                    name="wide_embedding",
                    initializer=fluid.initializer.Uniform()
                )
            )
            emb = paddle.reshape(emb, [-1, 1])
            return emb

        sparse_emb_seq = list(map(sparse_embedding_layer, self.sparse_input))
        dnn_out = paddle.concat(
            sparse_emb_seq + [self.dense_input], axis=1)

        for idx, dnn_fc_size in enumerate(self.hidden_size):
            fc_result = paddle.static.nn.fc(
                x=dnn_out,
                size=dnn_fc_size,
                activation='relu',
                name="deep_fc_{}".format(idx),
                weight_attr=paddle.ParamAttr(initializer=fluid.initializer.Normal(
                    scale=1.0 / math.sqrt(dnn_out.shape[1]))))
            dnn_out = fc_result

        deep_out = paddle.static.nn.fc(
            x=dnn_out,
            size=1,
            name="deep_fc_{}".format(idx),
            weight_attr=paddle.ParamAttr(initializer=fluid.initializer.TruncatedNormal(
                loc=0.0, scale=1.0))
        )

        wide_emb_out = list(map(wide_embedding_layer, self.wide_input))
        print(wide_emb_out)
        wide_out = fluid.layers.reduce_sum(paddle.concat(wide_emb_out, axis=1))

        self.predict = fluid.layers.elementwise_add(wide_out, deep_out)

        pred = fluid.layers.sigmoid(
            fluid.layers.clip(
                self.predict, min=-15.0, max=15.0),
            name="prediction")
        auc, batch_auc, auc_states = fluid.layers.auc(
            input=pred, label=fluid.layers.cast(
                x=self.label_input, dtype='int64'))

        if is_infer:
            self._infer_results["AUC"] = auc
            self._infer_results["BATCH_AUC"] = batch_auc
            return

        self._metrics["AUC"] = auc
        self._metrics["BATCH_AUC"] = batch_auc

        cost = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=self.predict, label=fluid.layers.cast(
                self.label_input, dtype='float32'))

        avg_cost = fluid.layers.reduce_mean(cost)
        self._cost = avg_cost

    def optimizer(self):
        optimizer = paddle.optimizer.Adam(self.learning_rate, lazy_mode=True)
        return optimizer

    def infer_net(self):
        pass
