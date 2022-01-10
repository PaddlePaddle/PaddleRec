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
import math
from net import MHCN


class DygraphModel():
    # define model
    def create_model(self, config):
        n_layers = config.get("hyper_parameters.n_layer")
        emb_size = config.get("hyper_parameters.num_factors")
        mhcn_model = MHCN(n_layers=n_layers, emb_size=emb_size, config=config)

        return mhcn_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data):
        # u_idx, v_idx, neg_idx = inputs

        user_input = paddle.to_tensor(batch_data[0].numpy().astype('int64')
                                      .reshape(-1, 1))
        item_input = paddle.to_tensor(batch_data[1].numpy().astype('int64')
                                      .reshape(-1, 1))
        neg_item_input = paddle.to_tensor(batch_data[2].numpy().astype('int64')
                                          .reshape(-1, 1))
        print("user input shape: ", paddle.shape(user_input))
        print("item input shape: ", paddle.shape(item_input))

        return [user_input, item_input, neg_item_input]

    # define loss function
    def create_loss(self, outputs):
        user_emb, pos_item_emb, neg_item_emb = outputs
        score = paddle.sum(paddle.multiply(user_emb, pos_item_emb),
                           1) - paddle.sum(
                               paddle.multiply(user_emb, neg_item_emb), 1)
        loss = -paddle.sum(paddle.log(F.sigmoid(score) + 10e-8))
        return loss

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
        inputs = self.create_feeds(batch_data)

        prediction = dy_model.forward(inputs)
        loss = self.create_loss(prediction)
        # update metrics
        print_dict = {"loss": loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        inputs = self.create_feeds(batch_data)

        prediction = dy_model.forward(inputs)
        # update metrics
        # print_dict = {
        #     "user": inputs[0],
        # }
        return metrics_list, None
