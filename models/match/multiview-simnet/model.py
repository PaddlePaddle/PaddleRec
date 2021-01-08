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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.layers.tensor as tensor
import paddle.fluid.layers.control_flow as cf
from simnet_net import MultiviewSimnetLayer
from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


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
        self.query_len = envs.get_global_env("hyper_parameters.query_len")
        self.pos_len = envs.get_global_env("hyper_parameters.pos_len")
        self.neg_len = envs.get_global_env("hyper_parameters.neg_len")

    def net(self, input, is_infer=False):
        self.q_slots = self._sparse_data_var[0:1]
        self.pt_slots = self._sparse_data_var[1:2]
        if not is_infer:
            self.batch_size = envs.get_global_env(
                "dataset.dataset_train.batch_size")
            self.nt_slots = self._sparse_data_var[2:3]
            inputs = [self.q_slots, self.pt_slots, self.nt_slots]
        else:
            self.batch_size = envs.get_global_env(
                "dataset.dataset_infer.batch_size")
            inputs = [self.q_slots, self.pt_slots]
        simnet_model = MultiviewSimnetLayer(
            self.query_encoder, self.title_encoder, self.query_encode_dim,
            self.title_encode_dim, self.emb_size, self.emb_dim,
            self.hidden_size, self.margin, self.query_len, self.pos_len,
            self.neg_len)
        cos_pos, cos_neg = simnet_model(inputs, is_infer)

        if is_infer:
            self._infer_results['query_pt_sim'] = cos_pos
            return

        # pairwise hinge_loss
        loss_part1 = paddle.subtract(
            paddle.full(
                shape=[self.batch_size, 1],
                fill_value=self.margin,
                dtype='float32'),
            cos_pos)

        loss_part2 = paddle.add(loss_part1, cos_neg)

        loss_part3 = paddle.maximum(
            paddle.full(
                shape=[self.batch_size, 1], fill_value=0.0, dtype='float32'),
            loss_part2)

        self._cost = paddle.mean(loss_part3)
        self.acc = self.get_acc(cos_neg, cos_pos)
        self._metrics["loss"] = self._cost
        self._metrics["acc"] = self.acc

    def get_acc(self, x, y):
        less = paddle.cast(paddle.less_than(x, y), dtype='float32')
        label_ones = paddle.full(
            dtype='float32', shape=[self.batch_size, 1], fill_value=1.0)
        correct = paddle.sum(less)
        total = paddle.sum(label_ones)
        acc = paddle.divide(correct, total)
        return acc
