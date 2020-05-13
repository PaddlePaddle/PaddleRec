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

import numpy as np
import math
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.layers.tensor as tensor
import paddle.fluid.layers.control_flow as cf

from paddlerec.core.utils import envs
from paddlerec.core.model import Model as ModelBase

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
        self.init_config()
        
    def init_config(self):
        self._fetch_interval = 1 
        query_encoder = envs.get_global_env("hyper_parameters.query_encoder", None, self._namespace)
        title_encoder = envs.get_global_env("hyper_parameters.title_encoder", None, self._namespace)
        query_encode_dim = envs.get_global_env("hyper_parameters.query_encode_dim", None, self._namespace)
        title_encode_dim = envs.get_global_env("hyper_parameters.title_encode_dim", None, self._namespace)
        query_slots = envs.get_global_env("hyper_parameters.query_slots", None, self._namespace)
        title_slots = envs.get_global_env("hyper_parameters.title_slots", None, self._namespace)
        factory = SimpleEncoderFactory()
        self.query_encoders = [
            factory.create(query_encoder, query_encode_dim)
            for i in range(query_slots)
        ]
	self.title_encoders = [
            factory.create(title_encoder, title_encode_dim)
            for i in range(title_slots)
        ]

	self.emb_size = envs.get_global_env("hyper_parameters.sparse_feature_dim", None, self._namespace)
	self.emb_dim = envs.get_global_env("hyper_parameters.embedding_dim", None, self._namespace)
	self.emb_shape = [self.emb_size, self.emb_dim]
	self.hidden_size = envs.get_global_env("hyper_parameters.hidden_size", None, self._namespace)
	self.margin = 0.1

    def input(self, is_train=True):
	self.q_slots = [
            fluid.data(
                name="%d" % i, shape=[None, 1], lod_level=1, dtype='int64')
            for i in range(len(self.query_encoders))
        ]
        self.pt_slots = [
            fluid.data(
                name="%d" % (i + len(self.query_encoders)), shape=[None, 1], lod_level=1, dtype='int64')
            for i in range(len(self.title_encoders))
        ]

	if is_train == False:
	    return self.q_slots + self.pt_slots

        self.nt_slots = [
            fluid.data(
                name="%d" % (i + len(self.query_encoders) + len(self.title_encoders)), shape=[None, 1], lod_level=1, dtype='int64')
            for i in range(len(self.title_encoders))
        ]

        return self.q_slots + self.pt_slots + self.nt_slots
    
    def train_input(self):
        res = self.input()
        self._data_var = res

        use_dataloader = envs.get_global_env("hyper_parameters.use_DataLoader", False, self._namespace) 

        if self._platform != "LINUX" or use_dataloader:
            self._data_loader = fluid.io.DataLoader.from_generator(
                feed_list=self._data_var, capacity=256, use_double_buffer=False, iterable=False)

    def get_acc(self, x, y):
        less = tensor.cast(cf.less_than(x, y), dtype='float32')
	label_ones = fluid.layers.fill_constant_batch_size_like(
            input=x, dtype='float32', shape=[-1, 1], value=1.0)
        correct = fluid.layers.reduce_sum(less)
	total = fluid.layers.reduce_sum(label_ones)
        acc = fluid.layers.elementwise_div(correct, total)
	return acc

    def net(self):
	q_embs = [
            fluid.embedding(
                input=query, size=self.emb_shape, param_attr="emb")
            for query in self.q_slots
        ]
        pt_embs = [
            fluid.embedding(
                input=title, size=self.emb_shape, param_attr="emb")
            for title in self.pt_slots
        ]
        nt_embs = [
            fluid.embedding(
                input=title, size=self.emb_shape, param_attr="emb")
            for title in self.nt_slots
        ]
        
	# encode each embedding field with encoder
        q_encodes = [
            self.query_encoders[i].forward(emb) for i, emb in enumerate(q_embs)
        ]
        pt_encodes = [
            self.title_encoders[i].forward(emb) for i, emb in enumerate(pt_embs)
        ]
        nt_encodes = [
            self.title_encoders[i].forward(emb) for i, emb in enumerate(nt_embs)
        ]

        # concat multi view for query, pos_title, neg_title
        q_concat = fluid.layers.concat(q_encodes)
        pt_concat = fluid.layers.concat(pt_encodes)
        nt_concat = fluid.layers.concat(nt_encodes)

	# projection of hidden layer
        q_hid = fluid.layers.fc(q_concat,
                                size=self.hidden_size,
                                param_attr='q_fc.w',
                                bias_attr='q_fc.b')
        pt_hid = fluid.layers.fc(pt_concat,
                                 size=self.hidden_size,
                                 param_attr='t_fc.w',
                                 bias_attr='t_fc.b')
        nt_hid = fluid.layers.fc(nt_concat,
                                 size=self.hidden_size,
                                 param_attr='t_fc.w',
                                 bias_attr='t_fc.b')

        # cosine of hidden layers
        cos_pos = fluid.layers.cos_sim(q_hid, pt_hid)
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

        self.avg_cost = fluid.layers.mean(loss_part3)
       	self.acc = self.get_acc(cos_neg, cos_pos)	

    def avg_loss(self):
        self._cost = self.avg_cost

    def metrics(self):
        self._metrics["loss"] = self.avg_cost
        self._metrics["acc"] = self.acc

    def train_net(self):
        self.train_input()
        self.net()
        self.avg_loss()
        self.metrics()

    def optimizer(self):
        learning_rate = envs.get_global_env("hyper_parameters.learning_rate", None, self._namespace)
	optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
	return optimizer

    def infer_input(self):
        res = self.input(is_train=False)
	self._infer_data_var = res

        self._infer_data_loader = fluid.io.DataLoader.from_generator(
            feed_list=self._infer_data_var, capacity=64, use_double_buffer=False, iterable=False)
 
    def infer_net(self):
	self.infer_input()
	# lookup embedding for each slot
        q_embs = [
            fluid.embedding(
                input=query, size=self.emb_shape, param_attr="emb")
            for query in self.q_slots
        ]
        pt_embs = [
            fluid.embedding(
                input=title, size=self.emb_shape, param_attr="emb")
            for title in self.pt_slots
        ]
	# encode each embedding field with encoder
        q_encodes = [
            self.query_encoders[i].forward(emb) for i, emb in enumerate(q_embs)
        ]
        pt_encodes = [
            self.title_encoders[i].forward(emb) for i, emb in enumerate(pt_embs)
        ]
	# concat multi view for query, pos_title, neg_title
        q_concat = fluid.layers.concat(q_encodes)
        pt_concat = fluid.layers.concat(pt_encodes)
        # projection of hidden layer
        q_hid = fluid.layers.fc(q_concat,
                                size=self.hidden_size,
                                param_attr='q_fc.w',
                                bias_attr='q_fc.b')
        pt_hid = fluid.layers.fc(pt_concat,
                                 size=self.hidden_size,
                                 param_attr='t_fc.w',
                                 bias_attr='t_fc.b')

        # cosine of hidden layers
        cos = fluid.layers.cos_sim(q_hid, pt_hid)
        self._infer_results['query_pt_sim'] = cos
