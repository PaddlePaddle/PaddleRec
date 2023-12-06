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

import logging
import paddle
import paddle.nn as nn
from news_encoder import NewsEncoder
from user_encoder import UserEncoder
import math
import numpy as np

class FastRecommender(paddle.nn.Layer):
    def __init__(self, args):
        super(FastRecommender, self).__init__()

        self.args = args
        self.news_encoder = TextEncoder(args)
        self.user_encoder = UserEncoder(args, self.news_encoder if self.args.title_share_encoder else None)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,
                news_vecs,  # the index of embedding from cache
                hist_sequence,  # user num  history length
                hist_sequence_mask,  # user num  history length
                candidate_inx,  # user num  history length-1  npratio
                labels,
                compute_loss=True
                ):
        reshape_candidate = paddle.flatten(candidate_inx)
        candidate_vec = paddle.index_select(news_vecs, reshape_candidate, axis=0)
        candidate_vec = paddle.reshape(candidate_vec, shape=(candidate_inx.shape[0], candidate_inx.shape[1], -1))  # B N D

        reshape_hist = paddle.flatten(hist_sequence)
        log_vec = paddle.index_select(news_vecs, reshape_hist, axis=0)
        log_vec = paddle.reshape(log_vec, shape=(hist_sequence.shape[0], hist_sequence.shape[1], -1))  # batch_size, log_length, news_dim

        user_vec = self.user_encoder(
            log_vec, hist_sequence_mask, True
        ).unsqueeze(-1)

        score = paddle.matmul(candidate_vec, user_vec).squeeze(-1)
        if compute_loss:
            loss = self.loss_fn(score, labels)
            return loss, score
        else:
            return score

    def load_param(self, trained_path):
        param_dict = paddle.load(trained_path, map_location='cpu')['model_state_dict']
        for i in param_dict:
            if i not in self.state_dict().keys() or param_dict[i].shape != self.state_dict()[i.replace('module.', '')].shape:
                continue
            self.state_dict()[i.replace('module.', '')].set_value(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))
