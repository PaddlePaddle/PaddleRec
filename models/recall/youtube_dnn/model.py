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
import numpy as np


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def input_data(self, is_infer=False):

        watch_vec_size = envs.get_global_env("hyper_parameters.watch_vec_size",
                                             None, self._namespace)
        search_vec_size = envs.get_global_env(
            "hyper_parameters.search_vec_size", None, self._namespace)
        other_feat_size = envs.get_global_env(
            "hyper_parameters.other_feat_size", None, self._namespace)

        watch_vec = fluid.data(
            name="watch_vec", shape=[None, watch_vec_size], dtype="float32")
        search_vec = fluid.data(
            name="search_vec", shape=[None, search_vec_size], dtype="float32")
        other_feat = fluid.data(
            name="other_feat", shape=[None, other_feat_size], dtype="float32")
        label = fluid.data(name="label", shape=[None, 1], dtype="int64")
        inputs = [watch_vec] + [search_vec] + [other_feat] + [label]
        self._data_var = inputs

        return inputs

    def fc(self, tag, data, out_dim, active='relu'):
        init_stddev = 1.0
        scales = 1.0 / np.sqrt(data.shape[1])

        if tag == 'l4':
            p_attr = fluid.param_attr.ParamAttr(
                name='%s_weight' % tag,
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=init_stddev * scales))
        else:
            p_attr = None

        b_attr = fluid.ParamAttr(
            name='%s_bias' % tag, initializer=fluid.initializer.Constant(0.1))

        out = fluid.layers.fc(input=data,
                              size=out_dim,
                              act=active,
                              param_attr=p_attr,
                              bias_attr=b_attr,
                              name=tag)
        return out

    def net(self, inputs):
        output_size = envs.get_global_env("hyper_parameters.output_size", None,
                                          self._namespace)
        layers = envs.get_global_env("hyper_parameters.layers", None,
                                     self._namespace)
        concat_feats = fluid.layers.concat(input=inputs[:-1], axis=-1)

        l1 = self.fc('l1', concat_feats, layers[0], 'relu')
        l2 = self.fc('l2', l1, layers[1], 'relu')
        l3 = self.fc('l3', l2, layers[2], 'relu')
        l4 = self.fc('l4', l3, output_size, 'softmax')

        num_seqs = fluid.layers.create_tensor(dtype='int64')
        acc = fluid.layers.accuracy(input=l4, label=inputs[-1], total=num_seqs)

        cost = fluid.layers.cross_entropy(input=l4, label=inputs[-1])
        avg_cost = fluid.layers.mean(cost)

        self._cost = avg_cost
        self._metrics["acc"] = acc

    def train_net(self):
        input_data = self.input_data()
        self.net(input_data)

    def infer_net(self):
        pass
