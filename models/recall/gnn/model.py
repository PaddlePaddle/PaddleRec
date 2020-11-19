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
import paddle

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase
from paddlerec.core.metrics import RecallK


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.learning_rate = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate")
        self.decay_steps = envs.get_global_env(
            "hyper_parameters.optimizer.decay_steps")
        self.decay_rate = envs.get_global_env(
            "hyper_parameters.optimizer.decay_rate")
        self.l2 = envs.get_global_env("hyper_parameters.optimizer.l2")

        self.dict_size = envs.get_global_env(
            "hyper_parameters.sparse_feature_number")
        self.corpus_size = envs.get_global_env("hyper_parameters.corpus_size")

        self.train_batch_size = envs.get_global_env(
            "dataset.dataset_train.batch_size")
        self.evaluate_batch_size = envs.get_global_env(
            "dataset.dataset_infer.batch_size")

        self.hidden_size = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim")
        self.step = envs.get_global_env(
            "hyper_parameters.gnn_propogation_steps")

    def input_data(self, is_infer=False, **kwargs):
        if is_infer:
            bs = self.evaluate_batch_size
        else:
            bs = self.train_batch_size
        items = paddle.static.data(
            name="items", shape=[bs, -1],
            dtype="int64")  # [batch_size, uniq_max]
        seq_index = paddle.static.data(
            name="seq_index", shape=[bs, -1, 2],
            dtype="int32")  # [batch_size, seq_max, 2]
        last_index = paddle.static.data(
            name="last_index", shape=[bs, 2], dtype="int32")  # [batch_size, 2]
        adj_in = paddle.static.data(
            name="adj_in", shape=[bs, -1, -1],
            dtype="float32")  # [batch_size, seq_max, seq_max]
        adj_out = paddle.static.data(
            name="adj_out", shape=[bs, -1, -1],
            dtype="float32")  # [batch_size, seq_max, seq_max]
        mask = paddle.static.data(
            name="mask", shape=[bs, -1, 1],
            dtype="float32")  # [batch_size, seq_max, 1]
        label = paddle.static.data(
            name="label", shape=[bs, 1], dtype="int64")  # [batch_size, 1]

        res = [items, seq_index, last_index, adj_in, adj_out, mask, label]
        return res

    def net(self, inputs, is_infer=False):
        if is_infer:
            bs = self.evaluate_batch_size
        else:
            bs = self.train_batch_size

        stdv = 1.0 / math.sqrt(self.hidden_size)

        def embedding_layer(input,
                            table_name,
                            emb_dim,
                            initializer_instance=None):
            emb = paddle.static.nn.embedding(
                input=input,
                size=[self.dict_size, emb_dim],
                param_attr=paddle.ParamAttr(
                    name=table_name, initializer=initializer_instance))
            return emb

        sparse_initializer = paddle.fluid.initializer.Uniform(
            low=-stdv, high=stdv)
        items_emb = embedding_layer(inputs[0], "emb", self.hidden_size,
                                    sparse_initializer)
        pre_state = items_emb
        for i in range(self.step):
            pre_state = paddle.fluid.layers.nn.reshape(
                x=pre_state, shape=[bs, -1, self.hidden_size])
            state_in = paddle.static.nn.fc(
                x=pre_state,
                name="state_in",
                size=self.hidden_size,
                activation=None,
                num_flatten_dims=2,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.fluid.initializer.Uniform(
                        low=-stdv, high=stdv)),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.fluid.initializer.Uniform(
                        low=-stdv, high=stdv)))  # [batch_size, uniq_max, h]
            state_out = paddle.static.nn.fc(
                x=pre_state,
                name="state_out",
                size=self.hidden_size,
                activation=None,
                num_flatten_dims=2,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.fluid.initializer.Uniform(
                        low=-stdv, high=stdv)),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.fluid.initializer.Uniform(
                        low=-stdv, high=stdv)))  # [batch_size, uniq_max, h]

            state_adj_in = paddle.fluid.layers.matmul(
                inputs[3], state_in)  # [batch_size, uniq_max, h]
            state_adj_out = paddle.fluid.layers.matmul(
                inputs[4], state_out)  # [batch_size, uniq_max, h]

            gru_input = paddle.concat(x=[state_adj_in, state_adj_out], axis=2)

            gru_input = paddle.fluid.layers.nn.reshape(
                x=gru_input, shape=[-1, self.hidden_size * 2])
            gru_fc = paddle.static.nn.fc(x=gru_input,
                                         name="gru_fc",
                                         size=3 * self.hidden_size,
                                         bias_attr=False)
            pre_state, _, _ = paddle.fluid.layers.gru_unit(
                input=gru_fc,
                hidden=paddle.fluid.layers.nn.reshape(
                    x=pre_state, shape=[-1, self.hidden_size]),
                size=3 * self.hidden_size)

        final_state = paddle.fluid.layers.nn.reshape(
            pre_state, shape=[bs, -1, self.hidden_size])
        seq = paddle.gather_nd(x=final_state, index=inputs[1])
        last = paddle.gather_nd(x=final_state, index=inputs[2])

        seq_fc = paddle.static.nn.fc(
            x=seq,
            name="seq_fc",
            size=self.hidden_size,
            bias_attr=False,
            activation=None,
            num_flatten_dims=2,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.fluid.initializer.Uniform(
                    low=-stdv, high=stdv)))  # [batch_size, seq_max, h]
        last_fc = paddle.static.nn.fc(
            x=last,
            name="last_fc",
            size=self.hidden_size,
            bias_attr=False,
            activation=None,
            num_flatten_dims=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.fluid.initializer.Uniform(
                    low=-stdv, high=stdv)))  # [bathc_size, h]

        seq_fc_t = paddle.transpose(
            seq_fc, perm=[1, 0, 2])  # [seq_max, batch_size, h]
        add = paddle.add(x=seq_fc_t, y=last_fc)  # [seq_max, batch_size, h]
        b = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(
                value=0.0))  # [h]
        add = paddle.add(x=add, y=b)  # [seq_max, batch_size, h]

        add_sigmoid = paddle.nn.functional.sigmoid(
            add)  # [seq_max, batch_size, h]
        add_sigmoid = paddle.transpose(
            add_sigmoid, perm=[1, 0, 2])  # [batch_size, seq_max, h]

        weight = paddle.static.nn.fc(
            x=add_sigmoid,
            name="weight_fc",
            size=1,
            activation=None,
            num_flatten_dims=2,
            bias_attr=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.fluid.initializer.Uniform(
                    low=-stdv, high=stdv)))  # [batch_size, seq_max, 1]
        weight *= inputs[5]
        weight_mask = paddle.multiply(
            x=seq, y=weight, axis=0)  # [batch_size, seq_max, h]
        global_attention = paddle.sum(x=weight_mask, axis=1)  # [batch_size, h]

        final_attention = paddle.concat(
            x=[global_attention, last], axis=1)  # [batch_size, 2*h]
        final_attention_fc = paddle.static.nn.fc(
            x=final_attention,
            name="final_attention_fc",
            size=self.hidden_size,
            bias_attr=False,
            activation=None,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.fluid.initializer.Uniform(
                    low=-stdv, high=stdv)))  # [batch_size, h]

        all_vocab = np.arange(1, self.dict_size).reshape((-1)).astype('int32')
        all_vocab = paddle.cast(
            x=paddle.nn.functional.assign(all_vocab), dtype='int64')

        all_emb = paddle.static.nn.embedding(
            input=all_vocab,
            param_attr=paddle.ParamAttr(
                name="emb",
                initializer=paddle.fluid.initializer.Uniform(
                    low=-stdv, high=stdv)),
            size=[self.dict_size, self.hidden_size])  # [all_vocab, h]

        logits = paddle.fluid.layers.matmul(
            x=final_attention_fc, y=all_emb,
            transpose_y=True)  # [batch_size, all_vocab]
        softmax = paddle.nn.functional.softmax_with_cross_entropy(
            logits=logits, label=inputs[6])  # [batch_size, 1]
        self.loss = paddle.mean(x=softmax)  # [1]
        acc = RecallK(input=logits, label=inputs[6], k=20)
        self._cost = self.loss

        if is_infer:
            self._infer_results['P@20'] = acc
            self._infer_results['LOSS'] = self.loss
            return

        self._metrics["LOSS"] = self.loss
        self._metrics["Train_P@20"] = acc

    def optimizer(self):
        step_per_epoch = self.corpus_size // self.train_batch_size
        optimizer = paddle.optimizer.Adam(
            learning_rate=paddle.fluid.layers.exponential_decay(
                learning_rate=self.learning_rate,
                decay_steps=self.decay_steps * step_per_epoch,
                decay_rate=self.decay_rate),
            weight_decay=paddle.regularizer.L2Decay(coeff=self.l2))
        return optimizer
