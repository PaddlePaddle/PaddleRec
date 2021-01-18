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
from paddlerec.core.model import ModelBase
from paddlerec.core.utils import envs
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from tagspace_net import TagspaceLayer


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)
        self.cost = None
        self.metrics = {}
        self.vocab_text_size = envs.get_global_env(
            "hyper_parameters.vocab_text_size")
        self.vocab_tag_size = envs.get_global_env(
            "hyper_parameters.vocab_tag_size")
        self.emb_dim = envs.get_global_env("hyper_parameters.emb_dim")
        self.hid_dim = envs.get_global_env("hyper_parameters.hid_dim")
        self.win_size = envs.get_global_env("hyper_parameters.win_size")
        self.margin = envs.get_global_env("hyper_parameters.margin")
        self.neg_size = envs.get_global_env("hyper_parameters.neg_size")
        self.text_len = envs.get_global_env("hyper_parameters.text_len")

    def input_data(self, is_infer=False, **kwargs):
        text = paddle.static.data(
            name="text", shape=[None, 1], lod_level=1, dtype='int64')
        pos_tag = paddle.static.data(
            name="pos_tag", shape=[None, 1], lod_level=1, dtype='int64')
        neg_tag = paddle.static.data(
            name="neg_tag", shape=[None, 1], lod_level=1, dtype='int64')
        return [text, pos_tag, neg_tag]

    def net(self, input, is_infer=False):
        """ network"""
        if is_infer:
            self.batch_size = envs.get_global_env(
                "dataset.inferdata.batch_size")
        else:
            self.batch_size = envs.get_global_env(
                "dataset.sample_1.batch_size")
        tagspace_model = TagspaceLayer(
            self.vocab_text_size, self.vocab_tag_size, self.emb_dim,
            self.hid_dim, self.win_size, self.margin, self.neg_size,
            self.text_len)
        cos_pos, cos_neg = tagspace_model(input)
        # calculate hinge loss
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
        avg_cost = paddle.mean(loss_part3)

        less = paddle.cast(paddle.less_than(cos_neg, cos_pos), dtype='float32')
        label_ones = paddle.full(
            dtype='float32', shape=[self.batch_size, 1], fill_value=1.0)
        correct = paddle.sum(less)
        total = paddle.sum(label_ones)
        acc = paddle.divide(correct, total)
        self._cost = avg_cost

        if is_infer:
            self._infer_results["acc"] = acc
            self._infer_results["loss"] = self._cost
        else:
            self._metrics["acc"] = acc
            self._metrics["loss"] = self._cost
