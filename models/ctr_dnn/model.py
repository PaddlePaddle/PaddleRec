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

from fleet_rec.utils import envs


class Train(object):

    def __init__(self):
        self.sparse_inputs = []
        self.dense_input = None
        self.label_input = None

        self.sparse_input_varnames = []
        self.dense_input_varname = None
        self.label_input_varname = None
        
        self.namespace = "train.model"

    def input(self):
        def sparse_inputs():
            ids = envs.get_global_env("hyper_parameters.sparse_inputs_slots", None ,self.namespace)

            sparse_input_ids = [
                fluid.layers.data(name="C" + str(i),
                                  shape=[1],
                                  lod_level=1,
                                  dtype="int64") for i in range(1, ids)
            ]
            return sparse_input_ids, [var.name for var in sparse_input_ids]

        def dense_input():
            dim = envs.get_global_env("hyper_parameters.dense_input_dim", None ,self.namespace)

            dense_input_var = fluid.layers.data(name="dense_input",
                                                shape=[dim],
                                                dtype="float32")
            return dense_input_var, dense_input_var.name

        def label_input():
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            return label, label.name

        self.sparse_inputs, self.sparse_input_varnames = sparse_inputs()
        self.dense_input, self.dense_input_varname = dense_input()
        self.label_input, self.label_input_varname = label_input()

    def input_vars(self):
        return [self.dense_input] + self.sparse_inputs + [self.label_input]

    def input_varnames(self):
        return [input.name for input in self.input_vars()]

    def net(self):
        def embedding_layer(input):
            sparse_feature_number = envs.get_global_env("hyper_parameters.sparse_feature_number", None ,self.namespace)
            sparse_feature_dim = envs.get_global_env("hyper_parameters.sparse_feature_dim", None ,self.namespace)

            emb = fluid.layers.embedding(
                input=input,
                is_sparse=True,
                size=[sparse_feature_number, sparse_feature_dim],
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
        hidden_layers = envs.get_global_env("hyper_parameters.fc_sizes", None ,self.namespace)

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

    def avg_loss(self):
        cost = fluid.layers.cross_entropy(input=self.predict, label=self.label_input)
        avg_cost = fluid.layers.reduce_sum(cost)
        self.loss = avg_cost
        return avg_cost

    def metrics(self):
        auc, batch_auc, _ = fluid.layers.auc(input=self.predict,
                                             label=self.label_input,
                                             num_thresholds=2 ** 12,
                                             slide_steps=20)
        self.metrics = (auc, batch_auc)

        return self.metrics

    def metric_extras(self):
        self.metric_vars = [self.metrics[0]]
        self.metric_alias = ["AUC"]
        self.fetch_interval_batchs = 10 
        return (self.metric_vars, self.metric_alias, self.fetch_interval_batchs)

    def optimizer(self):
        learning_rate = envs.get_global_env("hyper_parameters.learning_rate", None ,self.namespace)
        optimizer = fluid.optimizer.Adam(learning_rate, lazy_mode=True)
        return optimizer


class Evaluate(object):
    def input(self):
        pass

    def net(self):
        pass
