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
import numpy as np
import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import Model as ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.item_len = envs.get_global_env("hyper_parameters.self.item_len",
                                            None, self._namespace)
        self.hidden_size = envs.get_global_env("hyper_parameters.hidden_size",
                                               None, self._namespace)
        self.user_vocab = envs.get_global_env("hyper_parameters.user_vocab",
                                              None, self._namespace)
        self.item_vocab = envs.get_global_env("hyper_parameters.item_vocab",
                                              None, self._namespace)
        self.embed_size = envs.get_global_env("hyper_parameters.embed_size",
                                              None, self._namespace)

    def input_data(self, is_infer=False):
        user_slot_names = fluid.data(
            name='user_slot_names',
            shape=[None, 1],
            dtype='int64',
            lod_level=1)
        item_slot_names = fluid.data(
            name='item_slot_names',
            shape=[None, self.item_len],
            dtype='int64',
            lod_level=1)
        lens = fluid.data(name='lens', shape=[None], dtype='int64')
        labels = fluid.data(
            name='labels',
            shape=[None, self.item_len],
            dtype='int64',
            lod_level=1)

        inputs = [user_slot_names] + [item_slot_names] + [lens] + [labels]

        # demo: hot to use is_infer:
        if is_infer:
            return inputs
        else:
            return inputs

    def net(self, inputs, is_infer=False):
        # user encode
        user_embedding = fluid.embedding(
            input=inputs[0],
            size=[self.user_vocab, self.embed_size],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(),
                regularizer=fluid.regularizer.L2Decay(1e-5)),
            is_sparse=True)

        user_feature = fluid.layers.fc(
            input=user_embedding,
            size=self.hidden_size,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormal(
                    loc=0.0, scale=np.sqrt(1.0 / self.hidden_size))),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.0)),
            act='relu',
            name='user_feature_fc')
        # item encode
        item_embedding = fluid.embedding(
            input=inputs[1],
            size=[self.item_vocab, self.embed_size],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(),
                regularizer=fluid.regularizer.L2Decay(1e-5)),
            is_sparse=True)

        item_embedding = fluid.layers.sequence_unpad(
            x=item_embedding, length=inputs[2])

        item_fc = fluid.layers.fc(
            input=item_embedding,
            size=self.hidden_size,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormal(
                    loc=0.0, scale=np.sqrt(1.0 / self.hidden_size))),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.0)),
            act='relu',
            name='item_fc')

        pos = self._fluid_sequence_get_pos(item_fc)
        pos_embed = fluid.embedding(
            input=pos,
            size=[self.user_vocab, self.embed_size],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(),
                regularizer=fluid.regularizer.L2Decay(1e-5)),
            is_sparse=True)

        pos_embed = fluid.layers.squeeze(pos_embed, [1])

        # item gru
        gru_input = fluid.layers.fc(
            input=fluid.layers.concat([item_fc, pos_embed], 1),
            size=self.hidden_size * 3,
            name='item_gru_fc')

        # forward gru
        item_gru_forward = fluid.layers.dynamic_gru(
            input=gru_input,
            size=self.hidden_size,
            is_reverse=False,
            h_0=user_feature)
        # backward gru
        item_gru_backward = fluid.layers.dynamic_gru(
            input=gru_input,
            size=self.hidden_size,
            is_reverse=True,
            h_0=user_feature)

        item_gru = fluid.layers.concat(
            [item_gru_forward, item_gru_backward], axis=1)

        out_click_fc1 = fluid.layers.fc(
            input=item_gru,
            size=self.hidden_size,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormal(
                    loc=0.0, scale=np.sqrt(1.0 / self.hidden_size))),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.0)),
            act='relu',
            name='out_click_fc1')

        click_prob = fluid.layers.fc(input=out_click_fc1,
                                     size=2,
                                     act='softmax',
                                     name='out_click_fc2')

        labels = fluid.layers.sequence_unpad(x=inputs[3], length=inputs[2])

        auc_val, batch_auc, auc_states = fluid.layers.auc(input=click_prob,
                                                          label=labels)

        if is_infer:
            self._infer_results["AUC"] = auc_val
            return

        loss = fluid.layers.reduce_mean(
            fluid.layers.cross_entropy(
                input=click_prob, label=labels))
        self._cost = loss
        self._metrics['auc'] = auc_val

    def _fluid_sequence_pad(self, input, pad_value, maxlen=None):
        """
        args:
            input: (batch*seq_len, dim)
        returns:
            (batch, max_seq_len, dim)
        """
        pad_value = fluid.layers.cast(
            fluid.layers.assign(input=np.array([pad_value], 'float32')),
            input.dtype)
        input_padded, _ = fluid.layers.sequence_pad(
            input, pad_value,
            maxlen=maxlen)  # (batch, max_seq_len, 1), (batch, 1)
        # TODO, maxlen=300, used to solve issues: https://github.com/PaddlePaddle/Paddle/issues/14164
        return input_padded

    def _fluid_sequence_get_pos(self, lodtensor):
        """
        args:
            lodtensor: lod = [[0,4,7]]
        return:
            pos: lod = [[0,4,7]]
                 data = [0,1,2,3,0,1,3]
                 shape = [-1, 1]
        """
        lodtensor = fluid.layers.reduce_sum(lodtensor, dim=1, keep_dim=True)
        assert lodtensor.shape == (-1, 1), (lodtensor.shape())
        ones = fluid.layers.cast(lodtensor * 0 + 1,
                                 'float32')  # (batch*seq_len, 1)
        ones_padded = self._fluid_sequence_pad(ones,
                                               0)  # (batch, max_seq_len, 1)
        ones_padded = fluid.layers.squeeze(ones_padded,
                                           [2])  # (batch, max_seq_len)
        seq_len = fluid.layers.cast(
            fluid.layers.reduce_sum(
                ones_padded, 1, keep_dim=True), 'int64')  # (batch, 1)
        seq_len = fluid.layers.squeeze(seq_len, [1])

        pos = fluid.layers.cast(
            fluid.layers.cumsum(
                ones_padded, 1, exclusive=True), 'int64')
        pos = fluid.layers.sequence_unpad(pos, seq_len)  # (batch*seq_len, 1)
        pos.stop_gradient = True
        return pos

    #def train_net(self):
    #    input_data = self.input_data()
    #    self.net(input_data)

    #def infer_net(self):
    #    input_data = self.input_data(is_infer=True)
    #    self.net(input_data, is_infer=True)
