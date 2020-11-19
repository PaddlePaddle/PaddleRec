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

from paddlerec.core.model import ModelBase
from paddlerec.core.utils import envs
import paddle


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
        text = input[0]
        pos_tag = input[1]
        neg_tag = input[2]

        text_emb = paddle.static.nn.embedding(
            input=text,
            size=[self.vocab_text_size, self.emb_dim],
            param_attr="text_emb")
        text_emb = paddle.squeeze(x=text_emb, axis=[1])
        pos_tag_emb = paddle.static.nn.embedding(
            input=pos_tag,
            size=[self.vocab_tag_size, self.emb_dim],
            param_attr="tag_emb")
        pos_tag_emb = paddle.squeeze(x=pos_tag_emb, axis=[1])
        neg_tag_emb = paddle.static.nn.embedding(
            input=neg_tag,
            size=[self.vocab_tag_size, self.emb_dim],
            param_attr="tag_emb")
        neg_tag_emb = paddle.squeeze(x=neg_tag_emb, axis=[1])

        conv_1d = paddle.fluid.nets.sequence_conv_pool(
            input=text_emb,
            num_filters=self.hid_dim,
            filter_size=self.win_size,
            act="tanh",
            pool_type="max",
            param_attr="cnn")
        text_hid = paddle.static.nn.fc(x=conv_1d,
                                       size=self.emb_dim,
                                       weight_attr="text_hid")
        cos_pos = paddle.fluid.layers.nn.cos_sim(pos_tag_emb, text_hid)
        mul_text_hid = paddle.fluid.layers.sequence_expand_as(
            x=text_hid, y=neg_tag_emb)
        mul_cos_neg = paddle.fluid.layers.nn.cos_sim(neg_tag_emb, mul_text_hid)
        cos_neg_all = paddle.fluid.layers.sequence_reshape(
            input=mul_cos_neg, new_dim=self.neg_size)
        # choose max negtive cosine
        cos_neg = paddle.max(x=cos_neg_all, axis=1, keepdim=True)
        # calculate hinge loss
        loss_part1 = paddle.fluid.layers.nn.elementwise_sub(
            paddle.fluid.layers.tensor.fill_constant_batch_size_like(
                input=cos_pos,
                shape=[-1, 1],
                value=self.margin,
                dtype='float32'),
            cos_pos)
        loss_part2 = paddle.add(x=loss_part1, y=cos_neg)
        loss_part3 = paddle.maximum(
            x=paddle.fluid.layers.tensor.fill_constant_batch_size_like(
                input=loss_part2, shape=[-1, 1], value=0.0, dtype='float32'),
            y=loss_part2)
        avg_cost = paddle.mean(x=loss_part3)

        less = paddle.cast(
            paddle.less_than(
                x=cos_neg, y=cos_pos), dtype='float32')
        label_ones = paddle.fluid.layers.fill_constant_batch_size_like(
            input=cos_neg, dtype='float32', shape=[-1, 1], value=1.0)
        correct = paddle.sum(x=less)
        total = paddle.sum(x=label_ones)
        acc = paddle.divide(x=correct, y=total)
        self._cost = avg_cost

        if is_infer:
            self._infer_results["acc"] = acc
            self._infer_results["loss"] = self._cost
        else:
            self._metrics["acc"] = acc
            self._metrics["loss"] = self._cost
