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

    def _init_hyper_parameters(self):
        self.recall_k = envs.get_global_env("hyper_parameters.recall_k")
        self.vocab_size = envs.get_global_env("hyper_parameters.vocab_size")
        self.hid_size = envs.get_global_env("hyper_parameters.hid_size")
        self.init_low_bound = envs.get_global_env(
            "hyper_parameters.init_low_bound")
        self.init_high_bound = envs.get_global_env(
            "hyper_parameters.init_high_bound")
        self.emb_lr_x = envs.get_global_env("hyper_parameters.emb_lr_x")
        self.gru_lr_x = envs.get_global_env("hyper_parameters.gru_lr_x")
        self.fc_lr_x = envs.get_global_env("hyper_parameters.fc_lr_x")

    def input_data(self, is_infer=False, **kwargs):

        # Input data
        src_wordseq = fluid.data(
            name="src_wordseq", shape=[None, 1], dtype="int64", lod_level=1)
        dst_wordseq = fluid.data(
            name="dst_wordseq", shape=[None, 1], dtype="int64", lod_level=1)

        return [src_wordseq, dst_wordseq]

    def net(self, inputs, is_infer=False):
        src_wordseq = inputs[0]
        dst_wordseq = inputs[1]

        emb = fluid.embedding(
            input=src_wordseq,
            size=[self.vocab_size, self.hid_size],
            param_attr=fluid.ParamAttr(
                name="emb",
                initializer=fluid.initializer.Uniform(
                    low=self.init_low_bound, high=self.init_high_bound),
                learning_rate=self.emb_lr_x),
            is_sparse=True)
        fc0 = fluid.layers.fc(input=emb,
                              size=self.hid_size * 3,
                              param_attr=fluid.ParamAttr(
                                  initializer=fluid.initializer.Uniform(
                                      low=self.init_low_bound,
                                      high=self.init_high_bound),
                                  learning_rate=self.gru_lr_x))
        gru_h0 = fluid.layers.dynamic_gru(
            input=fc0,
            size=self.hid_size,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=self.init_low_bound, high=self.init_high_bound),
                learning_rate=self.gru_lr_x))

        fc = fluid.layers.fc(input=gru_h0,
                             size=self.vocab_size,
                             act='softmax',
                             param_attr=fluid.ParamAttr(
                                 initializer=fluid.initializer.Uniform(
                                     low=self.init_low_bound,
                                     high=self.init_high_bound),
                                 learning_rate=self.fc_lr_x))
        cost = fluid.layers.cross_entropy(input=fc, label=dst_wordseq)
        acc = fluid.layers.accuracy(
            input=fc, label=dst_wordseq, k=self.recall_k)
        if is_infer:
            self._infer_results['recall20'] = acc
            return
        avg_cost = fluid.layers.mean(x=cost)

        self._cost = avg_cost
        self._metrics["cost"] = avg_cost
        self._metrics["acc"] = acc
