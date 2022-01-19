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
        query_encoder = config.get('hyper_parameters.query_encoder', "gru")
        title_encoder = config.get('hyper_parameters.title_encoder', "gru")
        query_encode_dim = config.get('hyper_parameters.query_encode_dim', 128)
        title_encode_dim = config.get('hyper_parameters.title_encode_dim', 128)
        emb_size = config.get('hyper_parameters.sparse_feature_dim', 6327)
        emb_dim = config.get('hyper_parameters.embedding_dim', 128)
        hidden_size = config.get('hyper_parameters.hidden_size', 128)
        margin = config.get('hyper_parameters.margin', 0.1)
        query_len = config.get('hyper_parameters.query_len', 79)
        pos_len = config.get('hyper_parameters.pos_len', 99)
        neg_len = config.get('hyper_parameters.neg_len', 90)

        simnet_model = net.MultiviewSimnetLayer(
            query_encoder, title_encoder, query_encode_dim, title_encode_dim,
            emb_size, emb_dim, hidden_size, margin, query_len, pos_len,
            neg_len)
        return simnet_model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds_train(self, batch_data, query_len, pos_len, neg_len):
        q_slots = [
            paddle.to_tensor(batch_data[0].numpy().astype('int64').reshape(
                -1, query_len))
        ]
        pt_slots = [
            paddle.to_tensor(batch_data[1].numpy().astype('int64').reshape(
                -1, pos_len))
        ]
        nt_slots = [
            paddle.to_tensor(batch_data[2].numpy().astype('int64').reshape(
                -1, neg_len))
        ]
        inputs = [q_slots, pt_slots, nt_slots]
        return inputs

    def create_feeds_infer(self, batch_data, query_len, pos_len):
        q_slots = [
            paddle.to_tensor(batch_data[0].numpy().astype('int64').reshape(
                -1, query_len))
        ]
        pt_slots = [
            paddle.to_tensor(batch_data[1].numpy().astype('int64').reshape(
                -1, pos_len))
        ]
        inputs = [q_slots, pt_slots]
        return inputs

    # define loss function by predicts and label
    def create_loss(self, batch_size, margin, cos_pos, cos_neg):
        loss_part1 = paddle.subtract(
            paddle.full(
                shape=[batch_size, 1], fill_value=margin, dtype='float32'),
            cos_pos)

        loss_part2 = paddle.add(loss_part1, cos_neg)

        loss_part3 = paddle.maximum(
            paddle.full(
                shape=[batch_size, 1], fill_value=0.0, dtype='float32'),
            loss_part2)

        avg_cost = paddle.mean(loss_part3)
        return avg_cost

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def get_acc(self, x, y, batch_size):
        less = paddle.cast(paddle.less_than(x, y), dtype='float32')
        label_ones = paddle.full(
            dtype='float32', shape=[batch_size, 1], fill_value=1.0)
        correct = paddle.sum(less)
        total = paddle.sum(label_ones)
        acc = paddle.divide(correct, total)
        return acc

    def create_metrics(self):
        metrics_list_name = []
        metrics_list = []
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        query_len = config.get('hyper_parameters.query_len', 79)
        pos_len = config.get('hyper_parameters.pos_len', 99)
        neg_len = config.get('hyper_parameters.neg_len', 90)
        margin = config.get('hyper_parameters.margin', 0.1)
        batch_size = config.get("runner.train_batch_size", 128)
        inputs = self.create_feeds_train(batch_data, query_len, pos_len,
                                         neg_len)

        cos_pos, cos_neg = dy_model.forward(inputs, False)
        loss = self.create_loss(batch_size, margin, cos_pos, cos_neg)
        # update metrics
        acc = self.get_acc(cos_neg, cos_pos, batch_size)
        print_dict = {"Acc": acc, "loss": loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        query_len = config.get('hyper_parameters.query_len', 79)
        pos_len = config.get('hyper_parameters.pos_len', 99)
        inputs = self.create_feeds_infer(batch_data, query_len, pos_len)
        cos_pos, cos_neg = dy_model.forward(inputs, True)
        # update metrics
        print_dict = {" query_pt_sim": cos_pos}
        return metrics_list, print_dict
