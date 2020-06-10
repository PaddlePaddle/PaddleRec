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
import paddle.fluid.layers.tensor as tensor
import paddle.fluid.layers.control_flow as cf

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


class BowEncoder(object):
    """ bow-encoder """

    def __init__(self):
        self.param_name = ""

    def forward(self, emb):
        return fluid.layers.sequence_pool(input=emb, pool_type='sum')


class CNNEncoder(object):
    """ cnn-encoder"""

    def __init__(self,
                 param_name="cnn",
                 win_size=3,
                 ksize=128,
                 act='tanh',
                 pool_type='max'):
        self.param_name = param_name
        self.win_size = win_size
        self.ksize = ksize
        self.act = act
        self.pool_type = pool_type

    def forward(self, emb):
        return fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=self.ksize,
            filter_size=self.win_size,
            act=self.act,
            pool_type=self.pool_type,
            param_attr=self.param_name + ".param",
            bias_attr=self.param_name + ".bias")


class GrnnEncoder(object):
    """ grnn-encoder """

    def __init__(self, param_name="grnn", hidden_size=128):
        self.param_name = param_name
        self.hidden_size = hidden_size

    def forward(self, emb):
        fc0 = fluid.layers.fc(input=emb,
                              size=self.hidden_size * 3,
                              param_attr=self.param_name + "_fc.w",
                              bias_attr=False)

        gru_h = fluid.layers.dynamic_gru(
            input=fc0,
            size=self.hidden_size,
            is_reverse=False,
            param_attr=self.param_name + ".param",
            bias_attr=self.param_name + ".bias")
        return fluid.layers.sequence_pool(input=gru_h, pool_type='max')


class SimpleEncoderFactory(object):
    def __init__(self):
        pass

    ''' create an encoder through create function '''

    def create(self, enc_type, enc_hid_size):
        if enc_type == "bow":
            bow_encode = BowEncoder()
            return bow_encode
        elif enc_type == "cnn":
            cnn_encode = CNNEncoder(ksize=enc_hid_size)
            return cnn_encode
        elif enc_type == "gru":
            rnn_encode = GrnnEncoder(hidden_size=enc_hid_size)
            return rnn_encode


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.query_encoder = envs.get_global_env(
            "hyper_parameters.query_encoder")
        self.title_encoder = envs.get_global_env(
            "hyper_parameters.title_encoder")
        self.query_encode_dim = envs.get_global_env(
            "hyper_parameters.query_encode_dim")
        self.title_encode_dim = envs.get_global_env(
            "hyper_parameters.title_encode_dim")

        self.emb_size = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim")
        self.emb_dim = envs.get_global_env("hyper_parameters.embedding_dim")
        self.emb_shape = [self.emb_size, self.emb_dim]

        self.hidden_size = envs.get_global_env("hyper_parameters.hidden_size")
        self.margin = envs.get_global_env("hyper_parameters.margin")

    def net(self, input, is_infer=False):
        factory = SimpleEncoderFactory()
        self.q_slots = self._sparse_data_var[0:1]
        self.query_encoders = [
            factory.create(self.query_encoder, self.query_encode_dim)
            for _ in self.q_slots
        ]
        q_embs = [
            fluid.embedding(
                input=query, size=self.emb_shape, param_attr="emb")
            for query in self.q_slots
        ]
        # encode each embedding field with encoder
        q_encodes = [
            self.query_encoders[i].forward(emb) for i, emb in enumerate(q_embs)
        ]
        # concat multi view for query, pos_title, neg_title
        q_concat = fluid.layers.concat(q_encodes)
        # projection of hidden layer
        q_hid = fluid.layers.fc(q_concat,
                                size=self.hidden_size,
                                param_attr='q_fc.w',
                                bias_attr='q_fc.b')

        self.pt_slots = self._sparse_data_var[1:2]
        self.title_encoders = [
            factory.create(self.title_encoder, self.title_encode_dim)
        ]
        pt_embs = [
            fluid.embedding(
                input=title, size=self.emb_shape, param_attr="emb")
            for title in self.pt_slots
        ]
        pt_encodes = [
            self.title_encoders[i].forward(emb)
            for i, emb in enumerate(pt_embs)
        ]
        pt_concat = fluid.layers.concat(pt_encodes)
        pt_hid = fluid.layers.fc(pt_concat,
                                 size=self.hidden_size,
                                 param_attr='t_fc.w',
                                 bias_attr='t_fc.b')
        # cosine of hidden layers
        cos_pos = fluid.layers.cos_sim(q_hid, pt_hid)

        if is_infer:
            self._infer_results['query_pt_sim'] = cos_pos
            return

        self.nt_slots = self._sparse_data_var[2:3]
        nt_embs = [
            fluid.embedding(
                input=title, size=self.emb_shape, param_attr="emb")
            for title in self.nt_slots
        ]
        nt_encodes = [
            self.title_encoders[i].forward(emb)
            for i, emb in enumerate(nt_embs)
        ]
        nt_concat = fluid.layers.concat(nt_encodes)
        nt_hid = fluid.layers.fc(nt_concat,
                                 size=self.hidden_size,
                                 param_attr='t_fc.w',
                                 bias_attr='t_fc.b')
        cos_neg = fluid.layers.cos_sim(q_hid, nt_hid)

        # pairwise hinge_loss
        loss_part1 = fluid.layers.elementwise_sub(
            tensor.fill_constant_batch_size_like(
                input=cos_pos,
                shape=[-1, 1],
                value=self.margin,
                dtype='float32'),
            cos_pos)

        loss_part2 = fluid.layers.elementwise_add(loss_part1, cos_neg)

        loss_part3 = fluid.layers.elementwise_max(
            tensor.fill_constant_batch_size_like(
                input=loss_part2, shape=[-1, 1], value=0.0, dtype='float32'),
            loss_part2)

        self._cost = fluid.layers.mean(loss_part3)
        self.acc = self.get_acc(cos_neg, cos_pos)
        self._metrics["loss"] = self._cost
        self._metrics["acc"] = self.acc

    def get_acc(self, x, y):
        less = tensor.cast(cf.less_than(x, y), dtype='float32')
        label_ones = fluid.layers.fill_constant_batch_size_like(
            input=x, dtype='float32', shape=[-1, 1], value=1.0)
        correct = fluid.layers.reduce_sum(less)
        total = fluid.layers.reduce_sum(label_ones)
        acc = fluid.layers.elementwise_div(correct, total)
        return acc
