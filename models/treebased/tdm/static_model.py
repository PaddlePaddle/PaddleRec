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
import numpy as np
from model import dnn_model_define


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate", 0.001)
        # model training hyper parameter
        self.node_nums = self.config.get("hyper_parameters.sparse_feature_num",
                                         10326150)
        self.x_bert_embed_size = self.config.get(
            "hyper_parameters.x_bert_embed_size", 128)
        self.node_emb_size = self.config.get("hyper_parameters.node_emb_size",
                                             24)
        self.item_nums = self.config.get("hyper_parameters.item_nums", 69)
        self.fea_group = self.config.get("hyper_parameters.fea_group",
                                         "20,20,10,10,2,2,2,1,1,1")
        self.with_att = self.config.get("hyper_parameters.with_att", False)

    def create_feeds(self, is_infer=False):
        '''
        user_input = [
            paddle.static.data(
                name="item_" + str(i + 1), shape=[None, 1], dtype="int64")
            for i in range(self.item_nums)
        ]
        '''
        user_input = paddle.static.data(
            name="item_id",
            shape=[None, self.x_bert_embed_size],
            dtype="float32")

        item = paddle.static.data(
            name="unit_id", shape=[None, 1], dtype="int64")

        label = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64")

        feed_list = [user_input, item, label]
        return feed_list

    def net(self, inputs, is_infer=False):

        pretrained_attr = paddle.ParamAttr(
            name='tdm.bw_emb.weight', trainable=True)
        embedding = paddle.nn.Embedding(
            self.node_nums,
            self.node_emb_size,
            sparse=True,
            padding_idx=0,
            weight_attr=pretrained_attr)

        #paddle.static.Print(inputs[i], summarize=-1)
        user_input_fc = paddle.static.nn.fc(
            x=inputs[0],
            size=self.x_bert_embed_size,
            activation="tanh",
            weight_attr=paddle.ParamAttr(
                name="user_fc_w",
                initializer=paddle.initializer.NormalInitializer(seed=1)),
            bias_attr=paddle.ParamAttr(
                name="user_fc_b",
                initializer=paddle.initializer.ConstantInitializer(value=0.1)))
        unit_id_emb = embedding(inputs[-2])
        dout = dnn_model_define(
            self.x_bert_embed_size,
            user_input_fc,
            unit_id_emb,
            node_emb_size=self.node_emb_size,
            fea_groups=self.fea_group,
            with_att=self.with_att)

        cost, softmax_prob = paddle.nn.functional.softmax_with_cross_entropy(
            logits=dout,
            label=inputs[-1],
            return_softmax=True,
            ignore_index=-1)
        ignore_label = paddle.full_like(inputs[-1], fill_value=-1)
        avg_cost = paddle.divide(
            paddle.sum(cost),
            paddle.sum(
                paddle.cast(
                    paddle.not_equal(inputs[-1], ignore_label),
                    dtype='float32')))

        self._cost = avg_cost
        '''
        auc, _, _ = paddle.static.auc(input=softmax_prob,
                                      label=inputs[-1],
                                      slide_steps=0)
        '''
        auc = paddle.metric.accuracy(input=softmax_prob, label=inputs[-1])

        self.inference_target_var = softmax_prob

        fetch_dict = {'cost': avg_cost, 'acc': auc}
        #fetch_dict = {'cost': avg_cost}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adam(
            learning_rate=self.learning_rate, lazy_mode=True)

        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)
