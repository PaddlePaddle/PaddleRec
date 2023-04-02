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
        vocab_text_size = config.get("hyper_parameters.vocab_text_size")
        vocab_tag_size = config.get("hyper_parameters.vocab_tag_size")
        emb_dim = config.get("hyper_parameters.emb_dim")
        hid_dim = config.get("hyper_parameters.hid_dim")
        win_size = config.get("hyper_parameters.win_size")
        margin = config.get("hyper_parameters.margin")
        neg_size = config.get("hyper_parameters.neg_size")
        text_len = config.get("hyper_parameters.text_len")

        tagspace_model = net.TagspaceLayer(vocab_text_size, vocab_tag_size,
                                           emb_dim, hid_dim, win_size, margin,
                                           neg_size, text_len)
        return tagspace_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, text_len, neg_size):
        text = paddle.to_tensor(batch_data[0].numpy().astype('int64').reshape(
            -1, text_len))
        pos_tag = paddle.to_tensor(batch_data[1].numpy().astype('int64')
                                   .reshape(-1, 1))
        neg_tag = paddle.to_tensor(batch_data[2].numpy().astype('int64')
                                   .reshape(-1, neg_size))
        return [text, pos_tag, neg_tag]

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
        optimizer = paddle.optimizer.Adagrad(
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
        neg_size = config.get("hyper_parameters.neg_size")
        text_len = config.get("hyper_parameters.text_len")
        margin = config.get("hyper_parameters.margin")
        batch_size = config.get("runner.train_batch_size", 128)
        inputs = self.create_feeds(batch_data, text_len, neg_size)

        cos_pos, cos_neg = dy_model.forward(inputs)
        loss = self.create_loss(batch_size, margin, cos_pos, cos_neg)
        # update metrics
        acc = self.get_acc(cos_neg, cos_pos, batch_size)
        print_dict = {"loss": loss, "ACC": acc}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        neg_size = config.get("hyper_parameters.neg_size")
        text_len = config.get("hyper_parameters.text_len")
        batch_size = config.get("runner.infer_batch_size", 128)
        inputs = self.create_feeds(batch_data, text_len, neg_size)

        cos_pos, cos_neg = dy_model.forward(inputs)
        # update metrics
        acc = self.get_acc(cos_neg, cos_pos, batch_size)
        print_dict = {"ACC": acc}
        return metrics_list, print_dict
