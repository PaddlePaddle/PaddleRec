# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import math
import sys
import os
tdm_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "tdm"))
sys.path.append(tdm_path)
from model import dnn_model_define
from paddle.static import Program


class UserPreferenceModel:
    def __init__(self, init_model_path, tree_node_num, node_emb_size):
        self.init_model_path = init_model_path
        self.place = paddle.CPUPlace()
        self.exe = paddle.static.Executor(self.place)

        self.node_emb_size = node_emb_size
        self.create_embedding_lookup_model(tree_node_num)
        self.create_prediction_model()

    def calc_prediction_weight(self, sample_set, paths):
        n_samples = len(sample_set)
        n_path = len(paths)

        user_emb = self.user_embedding_lookup(sample_set)
        user_emb = [
            np.repeat(
                user_emb[i], n_path, axis=0) for i in range(len(user_emb))
        ]

        node_emb = self.node_embedding_lookup(paths)
        node_emb = np.concatenate([node_emb] * n_samples, axis=0)

        prob = self.calc_prob(user_emb, node_emb)
        return np.sum(prob)

    def calc_prob(self, user_inputs, unit_id_emb):
        feed_dict = {}
        for i in range(69):
            feed_dict["user_emb_{}".format(i)] = user_inputs[i]
        feed_dict["unit_id_emb"] = unit_id_emb

        res = self.exe.run(self.prediction_model,
                           feed=feed_dict,
                           fetch_list=self.prediction_model_fetch_vars)
        return res[0]

    def node_embedding_lookup(self, all_nodes):
        """ embedding lookup 
        """
        all_nodes = np.array(all_nodes).reshape([-1, 1]).astype('int64')
        res = []
        res = self.exe.run(self.embedding_lookup_program,
                           feed={"all_nodes": all_nodes},
                           fetch_list=self.embedding_fetch_var_names)
        return np.expand_dims(res[0], axis=1)

    def user_embedding_lookup(self, user_ids):
        all_nodes = np.array(user_ids).astype('int64')
        shape = all_nodes.shape
        if (shape[-1] != 1):
            shape = list(shape) + [1]
        all_nodes = all_nodes.reshape(shape)

        res = []
        res = self.exe.run(self.embedding_lookup_program,
                           feed={"all_nodes": all_nodes},
                           fetch_list=self.embedding_fetch_var_names)

        user_embeddings = []
        for i in range(all_nodes.shape[1]):
            user_embeddings.append(np.expand_dims(res[0][:, i, :], axis=1))
        return user_embeddings

    def create_embedding_lookup_model(self, tree_node_num):
        self.embedding_lookup_program = Program()
        startup = Program()

        with paddle.static.framework.program_guard(
                self.embedding_lookup_program, startup):
            all_nodes = paddle.static.data(
                name="all_nodes",
                shape=[-1, 1],
                dtype="int64",
                lod_level=1, )

            output = paddle.static.nn.embedding(
                input=all_nodes,
                is_sparse=True,
                size=[tree_node_num, self.node_emb_size],
                param_attr=paddle.ParamAttr(
                    name="tdm.bw_emb.weight",
                    initializer=paddle.initializer.UniformInitializer()))

            self.embedding_fetch_var_names = [output.name]
            self.exe.run(startup)

    def create_prediction_model(self, with_att=False):
        self.prediction_model = Program()
        startup = Program()

        with paddle.static.program_guard(self.prediction_model, startup):
            user_input = [
                paddle.static.data(
                    name="user_emb_{}".format(i),
                    shape=[-1, 1, self.node_emb_size],
                    dtype="float32", ) for i in range(69)
            ]
            unit_id_emb = paddle.static.data(
                name="unit_id_emb",
                shape=[-1, 1, self.node_emb_size],
                dtype="float32")
            dout = dnn_model_define(
                user_input, unit_id_emb, self.node_emb_size, with_att=with_att)
            softmax_prob = paddle.nn.functional.softmax(dout)
            positive_prob = paddle.slice(
                softmax_prob, axes=[1], starts=[1], ends=[2])
            prob = paddle.reshape(positive_prob, [-1])
            #print(str(self.prediction_model))
            self.prediction_model_fetch_vars = [prob.name]
            self.exe.run(startup)
