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
import paddle

from net import AutoInt


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.is_distributed = False
        self.distributed_embedding = False

        if self.config.get("hyper_parameters.distributed_embedding", 0) == 1:
            self.distributed_embedding = True

        self.feature_number = self.config.get("hyper_parameters.feature_number")
        self.embedding_dim = self.config.get("hyper_parameters.embedding_dim")
        self.fc_sizes = self.config.get("hyper_parameters.fc_sizes")
        self.use_residual = self.config.get("hyper_parameters.use_residual")
        self.scaling = self.config.get("hyper_parameters.scaling")
        self.use_wide = self.config.get("hyper_parameters.use_wide")
        self.use_sparse = self.config.get("hyper_parameters.use_sparse")
        self.head_num = self.config.get("hyper_parameters.head_num")
        self.num_field = self.config.get("hyper_parameters.num_field")
        self.attn_layer_sizes = self.config.get("hyper_parameters.attn_layer_sizes")
        self.learning_rate = self.config.get("hyper_parameters.optimizer.learning_rate")        


    def create_feeds(self, is_infer=False):
        self.label_input = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64")

        self.feat_index = paddle.static.data(
            name='feat_index',
            shape=[None, self.num_field],
            dtype='int64')

        self.feat_value = paddle.static.data(
            name='feat_value',
            shape=[None, self.num_field],
            dtype='float32'
        )

        feeds_list = [self.label_input, self.feat_index, self.feat_value]
        return feeds_list

    def net(self, input, is_infer=False):
        autoint_model = AutoInt(
            self.feature_number, self.embedding_dim, self.fc_sizes, self.use_residual, self.scaling, self.use_wide, 
            self.use_sparse, self.head_num, self.num_field, self.attn_layer_sizes)

        pred = autoint_model.forward(self.feat_index, self.feat_value)

        #pred = F.sigmoid(prediction)

        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)

        auc, batch_auc_var, _ = paddle.static.auc(input=predict_2d,
                                                  label=self.label_input,
                                                  slide_steps=0)

        self.inference_target_var = auc
        if is_infer:
            fetch_dict = {'auc': auc}
            return fetch_dict

        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                self.label_input, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        self._cost = avg_cost
        fetch_dict = {'cost': avg_cost, 'auc': auc}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adam(
            learning_rate=self.learning_rate, lazy_mode=True)
        if strategy != None:
            optimizer = paddle.distributed.fleet.distributed_optimizer(
                optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)
