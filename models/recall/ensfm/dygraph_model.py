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
import net


class DygraphModel():
    # define model
    def create_model(self, config):
        num_users = config.get("hyper_parameters.num_users")
        num_items = config.get("hyper_parameters.num_items")
        mf_dim = config.get("hyper_parameters.mf_dim")
        enc_fm_model = net.ENSFMLayer(num_users, num_items, mf_dim)
        return enc_fm_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data):
        batch_data[1] = batch_data[1][0]
        if len(batch_data) == 4:
            batch_data[3] = batch_data[3][0]

        return [paddle.to_tensor(x.numpy()) for x in batch_data]

    # define loss function by predicts and label
    def create_loss(self, prediction, config):
        pre, pos_r, q_emb, p_emb, H_i_emb = prediction
        weight = config.get('hyper_parameters.negative_weight', 0.5)
        loss = weight * paddle.sum(paddle.sum(
            paddle.sum(paddle.einsum('ab,ac->abc', q_emb, q_emb), 0) *
            paddle.sum(paddle.einsum('ab,ac->abc', p_emb, p_emb), 0) *
            paddle.matmul(
                H_i_emb, H_i_emb, transpose_y=True),
            0),
                                   0)
        loss += paddle.sum((1.0 - weight) * paddle.square(pos_r) - 2.0 * pos_r)
        return loss

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.05)
        optimizer = paddle.optimizer.Adagrad(
            learning_rate=lr,
            initial_accumulator_value=1e-8,
            parameters=dy_model.parameters())
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

        prediction = dy_model.forward(*inputs)
        loss = self.create_loss(prediction, config)
        # update metrics
        print_dict = {"loss": loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        inputs = self.create_feeds(batch_data)

        prediction = dy_model.forward(*inputs)
        # update metrics
        print_dict = {
            "user": inputs[0],
            "prediction": prediction[0],
        }
        return metrics_list, print_dict
