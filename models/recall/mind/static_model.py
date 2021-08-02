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
import net
import numpy as np


class StaticModel():
    """StaticModel
    """

    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters(config)

    def _init_hyper_parameters(self, config):
        self.item_count = config.get("hyper_parameters.item_count", None)
        self.embedding_dim = config.get("hyper_parameters.embedding_dim", 64)
        self.hidden_size = config.get("hyper_parameters.hidden_size", 64)
        self.neg_samples = config.get("hyper_parameters.neg_samples", 100)
        self.maxlen = config.get("hyper_parameters.maxlen", 30)
        self.pow_p = config.get("hyper_parameters.pow_p", 1.0)
        self.capsual_iters = config.get("hyper_parameters.capsual.iters", 3)
        self.capsual_max_k = config.get("hyper_parameters.capsual.max_k", 4)
        self.capsual_init_std = config.get("hyper_parameters.capsual.init_std",
                                           1.0)
        self.lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, is_infer=False):
        # print(batch_data)
        if not is_infer:
            hist_item = paddle.static.data(
                name="hist_item", shape=[-1, self.maxlen], dtype="int64")
            target_item = paddle.static.data(
                name="target_item", shape=[-1, 1], dtype="int64")
            seq_len = paddle.static.data(
                name="seq_len", shape=[-1, 1], dtype="int64")
            return [hist_item, target_item, seq_len]
        else:
            hist_item = paddle.static.data(
                name="hist_item", shape=[-1, self.maxlen], dtype="int64")
            seq_len = paddle.static.data(
                name="seq_len", shape=[-1, 1], dtype="int64")
            return [hist_item, seq_len]

    def net(self, inputs, is_infer=False):
        mind_model = net.MindLayer(self.item_count, self.embedding_dim,
                                   self.hidden_size, self.neg_samples,
                                   self.maxlen, self.pow_p, self.capsual_iters,
                                   self.capsual_max_k, self.capsual_init_std)
        # self.model = mind_model
        if is_infer:
            mind_model.eval()
            user_cap, cap_weights = mind_model(*inputs)
            # self.inference_target_var = user_cap
            fetch_dict = {"user_cap": user_cap}
            return fetch_dict

        hist_item, labels, seqlen = inputs
        [_, sampled_logist,
         sampled_labels], weight, user_cap, cap_weights, cap_mask = mind_model(
             hist_item, seqlen, labels)

        loss = F.softmax_with_cross_entropy(
            sampled_logist, sampled_labels, soft_label=True)
        self._cost = paddle.mean(loss)
        fetch_dict = {"loss": self._cost}
        return fetch_dict

    # define optimizer 
    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adam(learning_rate=self.lr)
        optimizer.minimize(self._cost)

    # construct infer forward phase  
    def infer_net(self, inputs):
        return self.net(inputs, is_infer=True)
