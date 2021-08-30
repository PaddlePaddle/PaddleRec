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


class DygraphModel():
    """DygraphModel
    """

    def create_model(self, config):
        item_count = config.get("hyper_parameters.item_count", None)
        embedding_dim = config.get("hyper_parameters.embedding_dim", 64)
        hidden_size = config.get("hyper_parameters.hidden_size", 64)
        neg_samples = config.get("hyper_parameters.neg_samples", 100)
        maxlen = config.get("hyper_parameters.maxlen", 30)
        pow_p = config.get("hyper_parameters.pow_p", 1.0)
        capsual_iters = config.get("hyper_parameters.capsual.iters", 3)
        capsual_max_k = config.get("hyper_parameters.capsual.max_k", 4)
        capsual_init_std = config.get("hyper_parameters.capsual.init_std", 1.0)
        MIND_model = net.MindLayer(item_count, embedding_dim, hidden_size,
                                   neg_samples, maxlen, pow_p, capsual_iters,
                                   capsual_max_k, capsual_init_std)
        return MIND_model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds_train(self, batch_data):
        #print(batch_data)
        hist_item = paddle.to_tensor(batch_data[0], dtype="int64")
        target_item = paddle.to_tensor(batch_data[1], dtype="int64")
        seq_len = paddle.to_tensor(batch_data[2], dtype="int64")
        return [hist_item, target_item, seq_len]

    #create_feeds_infer
    def create_feeds_infer(self, batch_data):
        batch_size = batch_data[0].shape[0]
        hist_item = paddle.to_tensor(batch_data[0], dtype="int64")
        target_item = paddle.zeros((batch_size, 1), dtype="int64")
        seq_len = paddle.to_tensor(batch_data[1], dtype="int64")
        return [hist_item, target_item, seq_len]

    # define optimizer 
    def create_loss(self, hit_prob):
        return paddle.mean(hit_prob)

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = []
        metrics_list = []
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        hist_item, labels, seqlen = self.create_feeds_train(batch_data)
        [loss, sampled_logist, sampled_labels
         ], weight, _, _, _ = dy_model.forward(hist_item, seqlen, labels)
        loss = self.create_loss(loss)
        print_dict = {"loss": loss}
        return loss, metrics_list, print_dict

    # construct infer forward phase  
    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        hist_item, labels, seqlen = self.create_feeds_infer(batch_data)
        dy_model.eval()
        user_cap, cap_weight = dy_model.forward(hist_item, seqlen, labels)
        # update metrics
        print_dict = None
        return user_cap, cap_weight
