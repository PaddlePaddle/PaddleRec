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

    def _init_hyper_parameters(self):
        self.is_distributed = True if envs.get_trainer(
        ) == "CtrTrainer" else False
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim")
        self.neg_num = envs.get_global_env("hyper_parameters.neg_num")
        self.with_shuffle_batch = envs.get_global_env(
            "hyper_parameters.with_shuffle_batch")
        self.learning_rate = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate")
        self.decay_steps = envs.get_global_env(
            "hyper_parameters.optimizer.decay_steps")
        self.decay_rate = envs.get_global_env(
            "hyper_parameters.optimizer.decay_rate")

    def input_data(self, is_infer=False, **kwargs):
        if is_infer:
            analogy_a = fluid.data(
                name="analogy_a", shape=[None, 1], lod_level=1, dtype='int64')
            analogy_b = fluid.data(
                name="analogy_b", shape=[None, 1], lod_level=1, dtype='int64')
            analogy_c = fluid.data(
                name="analogy_c", shape=[None, 1], lod_level=1, dtype='int64')
            analogy_d = fluid.data(
                name="analogy_d", shape=[None, 1], dtype='int64')
            return [analogy_a, analogy_b, analogy_c, analogy_d]

        input_word = fluid.data(
            name="input_word", shape=[None, 1], lod_level=1, dtype='int64')
        true_word = fluid.data(
            name='true_label', shape=[None, 1], lod_level=1, dtype='int64')
        if self.with_shuffle_batch:
            return [input_word, true_word]

        neg_word = fluid.data(
            name="neg_label", shape=[None, self.neg_num], dtype='int64')
        return [input_word, true_word, neg_word]

    def net(self, inputs, is_infer=False):
        if is_infer:
            self.infer_net(inputs)
            return

        def embedding_layer(input,
                            table_name,
                            initializer_instance=None,
                            sequence_pool=False):
            emb = fluid.embedding(
                input=input,
                is_sparse=True,
                is_distributed=self.is_distributed,
                size=[self.sparse_feature_number, self.sparse_feature_dim],
                param_attr=fluid.ParamAttr(
                    name=table_name, initializer=initializer_instance), )
            if sequence_pool:
                emb = fluid.layers.sequence_pool(
                    input=emb, pool_type='average')
            return emb

        init_width = 1.0 / self.sparse_feature_dim
        emb_initializer = fluid.initializer.Uniform(-init_width, init_width)
        emb_w_initializer = fluid.initializer.Constant(value=0.0)

        input_emb = embedding_layer(inputs[0], "emb", emb_initializer, True)
        input_emb = fluid.layers.squeeze(input=input_emb, axes=[1])
        true_emb_w = embedding_layer(inputs[1], "emb_w", emb_w_initializer,
                                     True)
        true_emb_w = fluid.layers.squeeze(input=true_emb_w, axes=[1])

        if self.with_shuffle_batch:
            neg_emb_w_list = []
            for i in range(self.neg_num):
                neg_emb_w_list.append(
                    fluid.contrib.layers.shuffle_batch(
                        true_emb_w))  # shuffle true_word
            neg_emb_w_concat = fluid.layers.concat(neg_emb_w_list, axis=0)
            neg_emb_w = fluid.layers.reshape(
                neg_emb_w_concat,
                shape=[-1, self.neg_num, self.sparse_feature_dim])
        else:
            neg_emb_w = embedding_layer(inputs[2], "emb_w", emb_w_initializer)
        true_logits = fluid.layers.reduce_sum(
            fluid.layers.elementwise_mul(input_emb, true_emb_w),
            dim=1,
            keep_dim=True)

        input_emb_re = fluid.layers.reshape(
            input_emb, shape=[-1, 1, self.sparse_feature_dim])
        neg_matmul = fluid.layers.matmul(
            input_emb_re, neg_emb_w, transpose_y=True)
        neg_logits = fluid.layers.reshape(neg_matmul, shape=[-1, 1])

        logits = fluid.layers.concat([true_logits, neg_logits], axis=0)
        label_ones = fluid.layers.fill_constant(
            shape=[fluid.layers.shape(true_logits)[0], 1],
            value=1.0,
            dtype='float32')
        label_zeros = fluid.layers.fill_constant(
            shape=[fluid.layers.shape(neg_logits)[0], 1],
            value=0.0,
            dtype='float32')
        label = fluid.layers.concat([label_ones, label_zeros], axis=0)

        loss = fluid.layers.log_loss(fluid.layers.sigmoid(logits), label)
        avg_cost = fluid.layers.reduce_sum(loss)
        self._cost = avg_cost
        self._metrics["LOSS"] = avg_cost

    def optimizer(self):
        optimizer = fluid.optimizer.SGD(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=self.learning_rate,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate,
                staircase=True))
        return optimizer

    def infer_net(self, inputs):
        def embedding_layer(input,
                            table_name,
                            initializer_instance=None,
                            sequence_pool=False):
            emb = fluid.embedding(
                input=input,
                size=[self.sparse_feature_number, self.sparse_feature_dim],
                param_attr=table_name)
            if sequence_pool:
                emb = fluid.layers.sequence_pool(
                    input=emb, pool_type='average')
            return emb

        all_label = np.arange(self.sparse_feature_number).reshape(
            self.sparse_feature_number).astype('int32')
        self.all_label = fluid.layers.cast(
            x=fluid.layers.assign(all_label), dtype='int64')
        emb_all_label = embedding_layer(self.all_label, "emb")
        fluid.layers.Print(inputs[0])
        fluid.layers.Print(inputs[1])
        fluid.layers.Print(inputs[2])
        fluid.layers.Print(inputs[3])
        emb_a = embedding_layer(inputs[0], "emb", sequence_pool=True)
        emb_b = embedding_layer(inputs[1], "emb", sequence_pool=True)
        emb_c = embedding_layer(inputs[2], "emb", sequence_pool=True)

        target = fluid.layers.elementwise_add(
            fluid.layers.elementwise_sub(emb_b, emb_a), emb_c)

        emb_all_label_l2 = fluid.layers.l2_normalize(x=emb_all_label, axis=1)
        dist = fluid.layers.matmul(
            x=target, y=emb_all_label_l2, transpose_y=True)
        values, pred_idx = fluid.layers.topk(input=dist, k=4)
        label = fluid.layers.expand(inputs[3], expand_times=[1, 4])
        label_ones = fluid.layers.fill_constant_batch_size_like(
            label, shape=[-1, 1], value=1.0, dtype='float32')
        right_cnt = fluid.layers.reduce_sum(input=fluid.layers.cast(
            fluid.layers.equal(pred_idx, label), dtype='float32'))
        total_cnt = fluid.layers.reduce_sum(label_ones)

        # global_right_cnt = fluid.layers.create_global_var(
        #     name="global_right_cnt",
        #     persistable=True,
        #     dtype='float32',
        #     shape=[1],
        #     value=0)
        # global_total_cnt = fluid.layers.create_global_var(
        #     name="global_total_cnt",
        #     persistable=True,
        #     dtype='float32',
        #     shape=[1],
        #     value=0)
        # global_right_cnt.stop_gradient = True
        # global_total_cnt.stop_gradient = True

        # tmp1 = fluid.layers.elementwise_add(right_cnt, global_right_cnt)
        # fluid.layers.assign(tmp1, global_right_cnt)
        # tmp2 = fluid.layers.elementwise_add(total_cnt, global_total_cnt)
        # fluid.layers.assign(tmp2, global_total_cnt)

        # acc = fluid.layers.elementwise_div(
        #     global_right_cnt, global_total_cnt, name="total_acc")
        acc = fluid.layers.elementwise_div(right_cnt, total_cnt, name="acc")
        self._infer_results['acc'] = acc
