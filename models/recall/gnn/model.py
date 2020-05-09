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

from fleetrec.core.utils import envs
from fleetrec.core.model import Model as ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)
        self.init_config()
        
    def init_config(self):
        self._fetch_interval = 1
        self.items_num, self.ins_num = self.config_read(envs.get_global_env("hyper_parameters.config_path", None, self._namespace))
        self.train_batch_size = envs.get_global_env("batch_size", None, "train.reader")
        self.evaluate_batch_size = envs.get_global_env("batch_size", None, "evaluate.reader")
        self.hidden_size = envs.get_global_env("hyper_parameters.sparse_feature_dim", None, self._namespace)
        self.step = envs.get_global_env("hyper_parameters.gnn_propogation_steps", None, self._namespace)


    def config_read(self, config_path=None):
	if config_path is None:
	    raise ValueError("please set train.model.hyper_parameters.config_path at first")
        with open(config_path, "r") as fin:
            item_nums = int(fin.readline().strip())
            ins_nums = int(fin.readline().strip())
        return item_nums, ins_nums

    def input(self, bs):
        self.items = fluid.data(
            name="items",
            shape=[bs, -1],
            dtype="int64") #[batch_size, uniq_max]
        self.seq_index = fluid.data(
            name="seq_index",
            shape=[bs, -1, 2],
            dtype="int32") #[batch_size, seq_max, 2]
        self.last_index = fluid.data(
            name="last_index",
            shape=[bs, 2],
            dtype="int32") #[batch_size, 2]
        self.adj_in = fluid.data(
            name="adj_in",
            shape=[bs, -1, -1],
            dtype="float32") #[batch_size, seq_max, seq_max]
        self.adj_out = fluid.data(
            name="adj_out",
            shape=[bs, -1, -1],
            dtype="float32") #[batch_size, seq_max, seq_max]
        self.mask = fluid.data(
            name="mask",
            shape=[bs, -1, 1],
            dtype="float32") #[batch_size, seq_max, 1]
        self.label = fluid.data(
            name="label",
            shape=[bs, 1],
            dtype="int64") #[batch_size, 1] 

        res = [self.items, self.seq_index, self.last_index, self.adj_in, self.adj_out, self.mask, self.label]
        return res
    
    def train_input(self):
        res = self.input(self.train_batch_size)
        self._data_var = res

        use_dataloader = envs.get_global_env("hyper_parameters.use_DataLoader", False, self._namespace) 

        if self._platform != "LINUX" or use_dataloader:
            self._data_loader = fluid.io.DataLoader.from_generator(
                feed_list=self._data_var, capacity=256, use_double_buffer=False, iterable=False)

    def net(self, items_num, hidden_size, step, bs):
	stdv = 1.0 / math.sqrt(hidden_size)

	def embedding_layer(input, table_name, emb_dim, initializer_instance=None):
            emb = fluid.embedding(
                input=input,
                size=[items_num, emb_dim],
                param_attr=fluid.ParamAttr(
                    name=table_name,
                    initializer=initializer_instance),
            )
	    return emb
	
	sparse_initializer = fluid.initializer.Uniform(low=-stdv, high=stdv)
	items_emb = embedding_layer(self.items, "emb", hidden_size, sparse_initializer)
        pre_state = items_emb
        for i in range(step):
            pre_state = layers.reshape(x=pre_state, shape=[bs, -1, hidden_size])
            state_in = layers.fc(
                input=pre_state,
                name="state_in",
                size=hidden_size,
                act=None,
                num_flatten_dims=2,
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                    low=-stdv, high=stdv)),
                bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                    low=-stdv, high=stdv)))  #[batch_size, uniq_max, h]
            state_out = layers.fc(
                input=pre_state,
                name="state_out",
                size=hidden_size,
                act=None,
                num_flatten_dims=2,
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                    low=-stdv, high=stdv)),
                bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                    low=-stdv, high=stdv)))  #[batch_size, uniq_max, h]
    
            state_adj_in = layers.matmul(self.adj_in, state_in)  #[batch_size, uniq_max, h]
            state_adj_out = layers.matmul(self.adj_out, state_out)   #[batch_size, uniq_max, h]
    
            gru_input = layers.concat([state_adj_in, state_adj_out], axis=2)
    
            gru_input = layers.reshape(x=gru_input, shape=[-1, hidden_size * 2])
            gru_fc = layers.fc(
                input=gru_input,
                name="gru_fc",
                size=3 * hidden_size,
                bias_attr=False)
            pre_state, _, _ = fluid.layers.gru_unit(
                input=gru_fc,
                hidden=layers.reshape(x=pre_state, shape=[-1, hidden_size]),
                size=3 * hidden_size)
    
        final_state = layers.reshape(pre_state, shape=[bs, -1, hidden_size])
        seq = layers.gather_nd(final_state, self.seq_index)
        last = layers.gather_nd(final_state, self.last_index)
    
        seq_fc = layers.fc(
            input=seq,
            name="seq_fc",
            size=hidden_size,
            bias_attr=False,
            act=None,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)))  #[batch_size, seq_max, h]
        last_fc = layers.fc(
            input=last,
            name="last_fc",
            size=hidden_size,
            bias_attr=False,
            act=None,
            num_flatten_dims=1,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)))  #[bathc_size, h]
    
        seq_fc_t = layers.transpose(
            seq_fc, perm=[1, 0, 2])  #[seq_max, batch_size, h]
        add = layers.elementwise_add(
            seq_fc_t, last_fc)  #[seq_max, batch_size, h]
        b = layers.create_parameter(
            shape=[hidden_size],
            dtype='float32',
            default_initializer=fluid.initializer.Constant(value=0.0))  #[h]
        add = layers.elementwise_add(add, b)  #[seq_max, batch_size, h]
    
        add_sigmoid = layers.sigmoid(add) #[seq_max, batch_size, h] 
        add_sigmoid = layers.transpose(
            add_sigmoid, perm=[1, 0, 2])  #[batch_size, seq_max, h]
    
        weight = layers.fc(
            input=add_sigmoid,
            name="weight_fc",
            size=1,
            act=None,
            num_flatten_dims=2,
            bias_attr=False,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-stdv, high=stdv)))  #[batch_size, seq_max, 1]
        weight *= self.mask
        weight_mask = layers.elementwise_mul(seq, weight, axis=0) #[batch_size, seq_max, h]
        global_attention = layers.reduce_sum(weight_mask, dim=1) #[batch_size, h]
    
        final_attention = layers.concat(
            [global_attention, last], axis=1)  #[batch_size, 2*h]
        final_attention_fc = layers.fc(
            input=final_attention,
            name="final_attention_fc",
            size=hidden_size,
            bias_attr=False,
            act=None,
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-stdv, high=stdv)))  #[batch_size, h]
    
       # all_vocab = layers.create_global_var(
       #     shape=[items_num - 1],
       #     value=0,
       #     dtype="int64",
       #     persistable=True,
       #     name="all_vocab")
        all_vocab = np.arange(1, items_num).reshape((-1)).astype('int32')
        all_vocab = fluid.layers.cast(x=fluid.layers.assign(all_vocab), dtype='int64')

        all_emb = fluid.embedding(
            input=all_vocab,
            param_attr=fluid.ParamAttr(
                name="emb",
                initializer=fluid.initializer.Uniform(
                    low=-stdv, high=stdv)),
            size=[items_num, hidden_size])  #[all_vocab, h]
    
        logits = layers.matmul(
            x=final_attention_fc, y=all_emb,
            transpose_y=True)  #[batch_size, all_vocab]
        softmax = layers.softmax_with_cross_entropy(
            logits=logits, label=self.label)  #[batch_size, 1]
        self.loss = layers.reduce_mean(softmax)  # [1]
        self.acc = layers.accuracy(input=logits, label=self.label, k=20)

    def avg_loss(self):
        self._cost = self.loss

    def metrics(self):
        self._metrics["LOSS"] = self.loss
        self._metrics["train_acc"] = self.acc

    def train_net(self):
        self.train_input()
        self.net(self.items_num, self.hidden_size, self.step, self.train_batch_size)
        self.avg_loss()
        self.metrics()

    def optimizer(self):
        learning_rate = envs.get_global_env("hyper_parameters.learning_rate", None, self._namespace)
        step_per_epoch = self.ins_num // self.train_batch_size
        decay_steps = envs.get_global_env("hyper_parameters.decay_steps", None, self._namespace)
        decay_rate = envs.get_global_env("hyper_parameters.decay_rate", None, self._namespace)
        l2 = envs.get_global_env("hyper_parameters.l2", None, self._namespace)
	optimizer = fluid.optimizer.Adam(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=learning_rate,
                decay_steps=decay_steps * step_per_epoch,
                decay_rate=decay_rate),
            regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=l2))

	return optimizer

    def infer_input(self):
        self._reader_namespace = "evaluate.reader"
        res = self.input(self.evaluate_batch_size)
	self._infer_data_var = res

        self._infer_data_loader = fluid.io.DataLoader.from_generator(
            feed_list=self._infer_data_var, capacity=64, use_double_buffer=False, iterable=False)
 
    def infer_net(self):
	self.infer_input()
	self.net(self.items_num, self.hidden_size, self.step, self.evaluate_batch_size)
        self._infer_results['acc'] = self.acc
	self._infer_results['loss'] = self.loss
