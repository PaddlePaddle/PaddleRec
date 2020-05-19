#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.layers.nn as nn
import paddle.fluid.layers.tensor as tensor
import paddle.fluid.layers.control_flow as cf

from paddlerec.core.model import Model as ModelBase
from paddlerec.core.utils import envs


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)
        self.cost = None
        self.metrics = {}
        self.vocab_text_size = envs.get_global_env("vocab_text_size", None, self._namespace)
        self.vocab_tag_size = envs.get_global_env("vocab_tag_size", None, self._namespace)
        self.emb_dim = envs.get_global_env("emb_dim", None, self._namespace)
        self.hid_dim = envs.get_global_env("hid_dim", None, self._namespace)
        self.win_size = envs.get_global_env("win_size", None, self._namespace)
        self.margin = envs.get_global_env("margin", None, self._namespace)
        self.neg_size = envs.get_global_env("neg_size", None, self._namespace)

    def train_net(self):
        """ network definition """
        text = fluid.data(name="text", shape=[None, 1], lod_level=1, dtype='int64')
        pos_tag = fluid.data(
            name="pos_tag", shape=[None, 1], lod_level=1, dtype='int64')
        neg_tag = fluid.data(
            name="neg_tag", shape=[None, 1], lod_level=1, dtype='int64')

        self._data_var = [text, pos_tag, neg_tag]

        text_emb = fluid.embedding(
            input=text, size=[self.vocab_text_size, self.emb_dim], param_attr="text_emb")
        text_emb = fluid.layers.squeeze(input=text_emb, axes=[1])
        pos_tag_emb = fluid.embedding(
            input=pos_tag, size=[self.vocab_tag_size, self.emb_dim], param_attr="tag_emb")
        pos_tag_emb = fluid.layers.squeeze(input=pos_tag_emb, axes=[1])
        neg_tag_emb = fluid.embedding(
            input=neg_tag, size=[self.vocab_tag_size, self.emb_dim], param_attr="tag_emb")
        neg_tag_emb = fluid.layers.squeeze(input=neg_tag_emb, axes=[1])

        conv_1d = fluid.nets.sequence_conv_pool(
            input=text_emb,
            num_filters=self.hid_dim,
            filter_size=self.win_size,
            act="tanh",
            pool_type="max",
            param_attr="cnn")
        text_hid = fluid.layers.fc(input=conv_1d,
                                   size=self.emb_dim,
                                   param_attr="text_hid")
        cos_pos = nn.cos_sim(pos_tag_emb, text_hid)
        mul_text_hid = fluid.layers.sequence_expand_as(x=text_hid, y=neg_tag_emb)
        mul_cos_neg = nn.cos_sim(neg_tag_emb, mul_text_hid)
        cos_neg_all = fluid.layers.sequence_reshape(
            input=mul_cos_neg, new_dim=self.neg_size)
        #choose max negtive cosine
        cos_neg = nn.reduce_max(cos_neg_all, dim=1, keep_dim=True)
        #calculate hinge loss
        loss_part1 = nn.elementwise_sub(
            tensor.fill_constant_batch_size_like(
                input=cos_pos, shape=[-1, 1], value=self.margin, dtype='float32'),
            cos_pos)
        loss_part2 = nn.elementwise_add(loss_part1, cos_neg)
        loss_part3 = nn.elementwise_max(
            tensor.fill_constant_batch_size_like(
                input=loss_part2, shape=[-1, 1], value=0.0, dtype='float32'),
            loss_part2)
        avg_cost = nn.mean(loss_part3)
        less = tensor.cast(cf.less_than(cos_neg, cos_pos), dtype='float32')
        correct = nn.reduce_sum(less)
        self.cost = avg_cost
        
        self.metrics["correct"] = correct
        self.metrics["cos_pos"] = cos_pos

    def get_avg_cost(self):
        return self.cost

    def get_metrics(self):
        return self.metrics

    def optimizer(self):
        learning_rate = envs.get_global_env("hyper_parameters.base_lr", None, self._namespace)
        sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=learning_rate)
        return sgd_optimizer

    def infer_net(self, parameter_list):
        self.train_net()
