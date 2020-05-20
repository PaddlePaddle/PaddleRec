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
import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import Model as ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def input(self):
        neg_num = int(
            envs.get_global_env("hyper_parameters.neg_num", None,
                                self._namespace))
        self.input_word = fluid.data(
            name="input_word", shape=[None, 1], dtype='int64')
        self.true_word = fluid.data(
            name='true_label', shape=[None, 1], dtype='int64')
        self._data_var.append(self.input_word)
        self._data_var.append(self.true_word)
        with_shuffle_batch = bool(
            int(
                envs.get_global_env("hyper_parameters.with_shuffle_batch",
                                    None, self._namespace)))
        if not with_shuffle_batch:
            self.neg_word = fluid.data(
                name="neg_label", shape=[None, neg_num], dtype='int64')
            self._data_var.append(self.neg_word)

        if self._platform != "LINUX":
            self._data_loader = fluid.io.DataLoader.from_generator(
                feed_list=self._data_var,
                capacity=64,
                use_double_buffer=False,
                iterable=False)

    def net(self):
        is_distributed = True if envs.get_trainer() == "CtrTrainer" else False
        neg_num = int(
            envs.get_global_env("hyper_parameters.neg_num", None,
                                self._namespace))
        sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number", None, self._namespace)
        sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim", None, self._namespace)
        with_shuffle_batch = bool(
            int(
                envs.get_global_env("hyper_parameters.with_shuffle_batch",
                                    None, self._namespace)))

        def embedding_layer(input,
                            table_name,
                            emb_dim,
                            initializer_instance=None,
                            squeeze=False):
            emb = fluid.embedding(
                input=input,
                is_sparse=True,
                is_distributed=is_distributed,
                size=[sparse_feature_number, emb_dim],
                param_attr=fluid.ParamAttr(
                    name=table_name, initializer=initializer_instance), )
            if squeeze:
                return fluid.layers.squeeze(input=emb, axes=[1])
            else:
                return emb

        init_width = 0.5 / sparse_feature_dim
        emb_initializer = fluid.initializer.Uniform(-init_width, init_width)
        emb_w_initializer = fluid.initializer.Constant(value=0.0)

        input_emb = embedding_layer(self.input_word, "emb", sparse_feature_dim,
                                    emb_initializer, True)
        true_emb_w = embedding_layer(self.true_word, "emb_w",
                                     sparse_feature_dim, emb_w_initializer,
                                     True)
        true_emb_b = embedding_layer(self.true_word, "emb_b", 1,
                                     emb_w_initializer, True)

        if with_shuffle_batch:
            neg_emb_w_list = []
            for i in range(neg_num):
                neg_emb_w_list.append(
                    fluid.contrib.layers.shuffle_batch(
                        true_emb_w))  # shuffle true_word
            neg_emb_w_concat = fluid.layers.concat(neg_emb_w_list, axis=0)
            neg_emb_w = fluid.layers.reshape(
                neg_emb_w_concat, shape=[-1, neg_num, sparse_feature_dim])

            neg_emb_b_list = []
            for i in range(neg_num):
                neg_emb_b_list.append(
                    fluid.contrib.layers.shuffle_batch(
                        true_emb_b))  # shuffle true_word
            neg_emb_b = fluid.layers.concat(neg_emb_b_list, axis=0)
            neg_emb_b_vec = fluid.layers.reshape(
                neg_emb_b, shape=[-1, neg_num])

        else:
            neg_emb_w = embedding_layer(self.neg_word, "emb_w",
                                        sparse_feature_dim, emb_w_initializer)
            neg_emb_b = embedding_layer(self.neg_word, "emb_b", 1,
                                        emb_w_initializer)
            neg_emb_b_vec = fluid.layers.reshape(
                neg_emb_b, shape=[-1, neg_num])

        true_logits = fluid.layers.elementwise_add(
            fluid.layers.reduce_sum(
                fluid.layers.elementwise_mul(input_emb, true_emb_w),
                dim=1,
                keep_dim=True),
            true_emb_b)

        input_emb_re = fluid.layers.reshape(
            input_emb, shape=[-1, 1, sparse_feature_dim])
        neg_matmul = fluid.layers.matmul(
            input_emb_re, neg_emb_w, transpose_y=True)
        neg_logits = fluid.layers.elementwise_add(
            fluid.layers.reshape(
                neg_matmul, shape=[-1, neg_num]),
            neg_emb_b_vec)

        label_ones = fluid.layers.fill_constant_batch_size_like(
            true_logits, shape=[-1, 1], value=1.0, dtype='float32')
        label_zeros = fluid.layers.fill_constant_batch_size_like(
            true_logits, shape=[-1, neg_num], value=0.0, dtype='float32')

        true_xent = fluid.layers.sigmoid_cross_entropy_with_logits(true_logits,
                                                                   label_ones)
        neg_xent = fluid.layers.sigmoid_cross_entropy_with_logits(neg_logits,
                                                                  label_zeros)
        cost = fluid.layers.elementwise_add(
            fluid.layers.reduce_sum(
                true_xent, dim=1),
            fluid.layers.reduce_sum(
                neg_xent, dim=1))
        self.avg_cost = fluid.layers.reduce_mean(cost)
        global_right_cnt = fluid.layers.create_global_var(
            name="global_right_cnt",
            persistable=True,
            dtype='float32',
            shape=[1],
            value=0)
        global_total_cnt = fluid.layers.create_global_var(
            name="global_total_cnt",
            persistable=True,
            dtype='float32',
            shape=[1],
            value=0)
        global_right_cnt.stop_gradient = True
        global_total_cnt.stop_gradient = True

    def avg_loss(self):
        self._cost = self.avg_cost

    def metrics(self):
        self._metrics["LOSS"] = self.avg_cost

    def train_net(self):
        self.input()
        self.net()
        self.avg_loss()
        self.metrics()

    def optimizer(self):
        learning_rate = envs.get_global_env("hyper_parameters.learning_rate",
                                            None, self._namespace)
        decay_steps = envs.get_global_env("hyper_parameters.decay_steps", None,
                                          self._namespace)
        decay_rate = envs.get_global_env("hyper_parameters.decay_rate", None,
                                         self._namespace)
        optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True))
        return optimizer

    def analogy_input(self):
        sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number", None, self._namespace)
        self.analogy_a = fluid.data(
            name="analogy_a", shape=[None], dtype='int64')
        self.analogy_b = fluid.data(
            name="analogy_b", shape=[None], dtype='int64')
        self.analogy_c = fluid.data(
            name="analogy_c", shape=[None], dtype='int64')
        self.analogy_d = fluid.data(
            name="analogy_d", shape=[None], dtype='int64')
        self._infer_data_var = [
            self.analogy_a, self.analogy_b, self.analogy_c, self.analogy_d
        ]

        self._infer_data_loader = fluid.io.DataLoader.from_generator(
            feed_list=self._infer_data_var,
            capacity=64,
            use_double_buffer=False,
            iterable=False)

    def infer_net(self):
        sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim", None, self._namespace)
        sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number", None, self._namespace)

        def embedding_layer(input, table_name, initializer_instance=None):
            emb = fluid.embedding(
                input=input,
                size=[sparse_feature_number, sparse_feature_dim],
                param_attr=table_name)
            return emb

        self.analogy_input()
        all_label = np.arange(sparse_feature_number).reshape(
            sparse_feature_number).astype('int32')
        self.all_label = fluid.layers.cast(
            x=fluid.layers.assign(all_label), dtype='int64')
        emb_all_label = embedding_layer(self.all_label, "emb")
        emb_a = embedding_layer(self.analogy_a, "emb")
        emb_b = embedding_layer(self.analogy_b, "emb")
        emb_c = embedding_layer(self.analogy_c, "emb")

        target = fluid.layers.elementwise_add(
            fluid.layers.elementwise_sub(emb_b, emb_a), emb_c)

        emb_all_label_l2 = fluid.layers.l2_normalize(x=emb_all_label, axis=1)
        dist = fluid.layers.matmul(
            x=target, y=emb_all_label_l2, transpose_y=True)
        values, pred_idx = fluid.layers.topk(input=dist, k=4)
        label = fluid.layers.expand(
            fluid.layers.unsqueeze(
                self.analogy_d, axes=[1]),
            expand_times=[1, 4])
        label_ones = fluid.layers.fill_constant_batch_size_like(
            label, shape=[-1, 1], value=1.0, dtype='float32')
        right_cnt = fluid.layers.reduce_sum(input=fluid.layers.cast(
            fluid.layers.equal(pred_idx, label), dtype='float32'))
        total_cnt = fluid.layers.reduce_sum(label_ones)

        global_right_cnt = fluid.layers.create_global_var(
            name="global_right_cnt",
            persistable=True,
            dtype='float32',
            shape=[1],
            value=0)
        global_total_cnt = fluid.layers.create_global_var(
            name="global_total_cnt",
            persistable=True,
            dtype='float32',
            shape=[1],
            value=0)
        global_right_cnt.stop_gradient = True
        global_total_cnt.stop_gradient = True

        tmp1 = fluid.layers.elementwise_add(right_cnt, global_right_cnt)
        fluid.layers.assign(tmp1, global_right_cnt)
        tmp2 = fluid.layers.elementwise_add(total_cnt, global_total_cnt)
        fluid.layers.assign(tmp2, global_total_cnt)

        acc = fluid.layers.elementwise_div(
            global_right_cnt, global_total_cnt, name="total_acc")
        self._infer_results['acc'] = acc
