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

import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import Model as ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def input(self):
        TRIGRAM_D = envs.get_global_env("hyper_parameters.TRIGRAM_D", None,
                                        self._namespace)

        Neg = envs.get_global_env("hyper_parameters.NEG", None,
                                  self._namespace)

        self.query = fluid.data(
            name="query", shape=[-1, TRIGRAM_D], dtype='float32', lod_level=0)
        self.doc_pos = fluid.data(
            name="doc_pos",
            shape=[-1, TRIGRAM_D],
            dtype='float32',
            lod_level=0)
        self.doc_negs = [
            fluid.data(
                name="doc_neg_" + str(i),
                shape=[-1, TRIGRAM_D],
                dtype="float32",
                lod_level=0) for i in range(Neg)
        ]
        self._data_var.append(self.query)
        self._data_var.append(self.doc_pos)
        for input in self.doc_negs:
            self._data_var.append(input)

        if self._platform != "LINUX":
            self._data_loader = fluid.io.DataLoader.from_generator(
                feed_list=self._data_var,
                capacity=64,
                use_double_buffer=False,
                iterable=False)

    def net(self, is_infer=False):
        hidden_layers = envs.get_global_env("hyper_parameters.fc_sizes", None,
                                            self._namespace)
        hidden_acts = envs.get_global_env("hyper_parameters.fc_acts", None,
                                          self._namespace)

        def fc(data, hidden_layers, hidden_acts, names):
            fc_inputs = [data]
            for i in range(len(hidden_layers)):
                xavier = fluid.initializer.Xavier(
                    uniform=True,
                    fan_in=fc_inputs[-1].shape[1],
                    fan_out=hidden_layers[i])
                out = fluid.layers.fc(input=fc_inputs[-1],
                                      size=hidden_layers[i],
                                      act=hidden_acts[i],
                                      param_attr=xavier,
                                      bias_attr=xavier,
                                      name=names[i])
                fc_inputs.append(out)
            return fc_inputs[-1]

        query_fc = fc(self.query, hidden_layers, hidden_acts,
                      ['query_l1', 'query_l2', 'query_l3'])
        doc_pos_fc = fc(self.doc_pos, hidden_layers, hidden_acts,
                        ['doc_pos_l1', 'doc_pos_l2', 'doc_pos_l3'])
        self.R_Q_D_p = fluid.layers.cos_sim(query_fc, doc_pos_fc)

        if is_infer:
            return

        R_Q_D_ns = []
        for i, doc_neg in enumerate(self.doc_negs):
            doc_neg_fc_i = fc(doc_neg, hidden_layers, hidden_acts, [
                'doc_neg_l1_' + str(i), 'doc_neg_l2_' + str(i),
                'doc_neg_l3_' + str(i)
            ])
            R_Q_D_ns.append(fluid.layers.cos_sim(query_fc, doc_neg_fc_i))
        concat_Rs = fluid.layers.concat(
            input=[self.R_Q_D_p] + R_Q_D_ns, axis=-1)
        prob = fluid.layers.softmax(concat_Rs, axis=1)

        hit_prob = fluid.layers.slice(
            prob, axes=[0, 1], starts=[0, 0], ends=[4, 1])
        loss = -fluid.layers.reduce_sum(fluid.layers.log(hit_prob))
        self.avg_cost = fluid.layers.mean(x=loss)

    def infer_results(self):
        self._infer_results['query_doc_sim'] = self.R_Q_D_p

    def avg_loss(self):
        self._cost = self.avg_cost

    def metrics(self):
        self._metrics["LOSS"] = self.avg_cost

    def train_net(self):
        self.input()
        self.net(is_infer=False)
        self.avg_loss()
        self.metrics()

    def optimizer(self):
        learning_rate = envs.get_global_env("hyper_parameters.learning_rate",
                                            None, self._namespace)
        optimizer = fluid.optimizer.SGD(learning_rate)
        return optimizer

    def infer_input(self):
        TRIGRAM_D = envs.get_global_env("hyper_parameters.TRIGRAM_D", None,
                                        self._namespace)
        self.query = fluid.data(
            name="query", shape=[-1, TRIGRAM_D], dtype='float32', lod_level=0)
        self.doc_pos = fluid.data(
            name="doc_pos",
            shape=[-1, TRIGRAM_D],
            dtype='float32',
            lod_level=0)
        self._infer_data_var = [self.query, self.doc_pos]

        self._infer_data_loader = fluid.io.DataLoader.from_generator(
            feed_list=self._infer_data_var,
            capacity=64,
            use_double_buffer=False,
            iterable=False)

    def infer_net(self):
        self.infer_input()
        self.net(is_infer=True)
        self.infer_results()
