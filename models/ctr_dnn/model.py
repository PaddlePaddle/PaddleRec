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

from ...utils import envs


class Train(object):

    def __init__(self):
        self.sparse_inputs = []
        self.dense_input = None
        self.label_input = None

        self.sparse_input_varnames = []
        self.dense_input_varname = None
        self.label_input_varname = None

    def input(self):
        def sparse_inputs():
            ids = envs.get_global_env("sparse_inputs_counts")

            sparse_input_ids = [
                fluid.layers.data(name="C" + str(i),
                                  shape=[1],
                                  lod_level=1,
                                  dtype="int64") for i in range(ids)
            ]
            return sparse_input_ids, [var.name for var in sparse_input_ids]

        def dense_input():
            dense_input_dim = envs.get_global_env("dense_input_dim")

            dense_input_var = fluid.layers.data(name="dense_input",
                                                shape=dense_input_dim,
                                                dtype="float32")
            return dense_input_var, dense_input_var.name

        def label_input():
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            return label, label.name

        self.sparse_inputs, self.sparse_input_varnames = sparse_inputs()
        self.dense_input, self.dense_input_varname = dense_input()
        self.label_input, self.label_input_varname = label_input()

    def input_vars(self):
        return self.sparse_inputs + [self.dense_input] + [self.label_input]

    def input_varnames(self):
        return [input.name for input in self.input_vars()]

    def net(self):
        def embedding_layer(input):
            sparse_feature_number = envs.get_global_env("sparse_feature_number")
            sparse_feature_dim = envs.get_global_env("sparse_feature_dim")

            emb = fluid.layers.embedding(
                input=input,
                is_sparse=True,
                size=[{sparse_feature_number}, {sparse_feature_dim}],
                param_attr=fluid.ParamAttr(
                    name="SparseFeatFactors",
                    initializer=fluid.initializer.Uniform()),
            )
            emb_sum = fluid.layers.sequence_pool(
                input=emb, pool_type='sum')
            return emb_sum

        def fc(input, output_size):
            output = fluid.layers.fc(
                input=input, size=output_size,
                act='relu', param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Normal(
                        scale=1.0 / math.sqrt(input.shape[1]))))
            return output

        sparse_embed_seq = list(map(embedding_layer, self.sparse_inputs))
        concated = fluid.layers.concat(sparse_embed_seq + [self.dense_input], axis=1)

        fcs = [concated]
        hidden_layers = envs.get_global_env("fc_sizes")

        for size in hidden_layers:
            fcs.append(fc(fcs[-1], size))

        predict = fluid.layers.fc(
            input=fcs[-1],
            size=2,
            act="softmax",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(fcs[-1].shape[1]))),
        )

        self.predict = predict

    def avg_loss(self, predict):
        cost = fluid.layers.cross_entropy(input=predict, label=self.label_input)
        avg_cost = fluid.layers.reduce_sum(cost)
        self.loss = avg_cost
        return avg_cost

    def metrics(self):
        auc, batch_auc, _ = fluid.layers.auc(input=self.predict,
                                             label=self.label_input,
                                             num_thresholds=2 ** 12,
                                             slide_steps=20)
        self.metrics = (auc, batch_auc)

    def optimizer(self):
        learning_rate = envs.get_global_env("learning_rate")
        optimizer = fluid.optimizer.Adam(learning_rate, lazy_mode=True)
        return optimizer

    def optimize(self):
        optimizer = self.optimizer()
        optimizer.minimize(self.loss)


class Evaluate(object):
    def input(self):
        pass

    def net(self):
        pass
