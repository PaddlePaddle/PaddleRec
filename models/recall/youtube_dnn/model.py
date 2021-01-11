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
import paddle.nn.functional as F

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase
from youtube_dnn_net import YoutubeDNNLayer


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.watch_vec_size = envs.get_global_env(
            "hyper_parameters.watch_vec_size")
        self.search_vec_size = envs.get_global_env(
            "hyper_parameters.search_vec_size")
        self.other_feat_size = envs.get_global_env(
            "hyper_parameters.other_feat_size")
        assert self.watch_vec_size == self.search_vec_size == self.other_feat_size
        self.vector_dim = self.watch_vec_size
        self.output_size = envs.get_global_env("hyper_parameters.output_size")
        self.layer_sizes = envs.get_global_env("hyper_parameters.layer_sizes")

    def input_data(self, is_infer=False, **kwargs):

        watch_vec = paddle.static.data(
            name="watch_vec",
            shape=[None, self.watch_vec_size],
            dtype="float32")
        search_vec = paddle.static.data(
            name="search_vec",
            shape=[None, self.search_vec_size],
            dtype="float32")
        other_feat = paddle.static.data(
            name="other_feat",
            shape=[None, self.other_feat_size],
            dtype="float32")
        label = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64")
        inputs = [watch_vec] + [search_vec] + [other_feat] + [label]

        return inputs

    def net(self, inputs, is_infer=False):
        concat_feats = paddle.concat(x=inputs[:-1], axis=-1)

        youtube_dnn_model = YoutubeDNNLayer(self.vector_dim * 3,
                                            self.layer_sizes, self.output_size)

        predict, output_layer = youtube_dnn_model(concat_feats)

        print(output_layer)
        pred = F.softmax(predict)

        acc = paddle.metric.accuracy(input=pred, label=inputs[-1])

        cost = F.cross_entropy(input=pred, label=inputs[-1])
        avg_cost = paddle.mean(x=cost)

        self._cost = avg_cost
        self._metrics["acc"] = acc
