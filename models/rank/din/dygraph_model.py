# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
import numpy as np
import net


class DygraphModel():
    def __init__(self):
        self.bucket = 100000
        self.absolute_limt = 200.0

    def rescale(self, number):
        if number > self.absolute_limt:
            number = self.absolute_limt
        elif number < -self.absolute_limt:
            number = -self.absolute_limt
        return (number + self.absolute_limt) / (self.absolute_limt * 2 + 1e-8)

    def create_model(self, config):
        item_emb_size = config.get("hyper_parameters.item_emb_size", 64)
        cat_emb_size = config.get("hyper_parameters.cat_emb_size", 64)
        act = config.get("hyper_parameters.act", "sigmoid")
        is_sparse = config.get("hyper_parameters.is_sparse", False)
        use_DataLoader = config.get("hyper_parameters.use_DataLoader", False)
        item_count = config.get("hyper_parameters.item_count", 63001)
        cat_count = config.get("hyper_parameters.cat_count", 801)
        din_model = net.DINLayer(item_emb_size, cat_emb_size, act, is_sparse,
                                 use_DataLoader, item_count, cat_count)
        return din_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch, config):
        hist_item_seq = batch[0]
        hist_cat_seq = batch[1]
        target_item = batch[2]
        target_cat = batch[3]
        label = paddle.reshape(batch[4], [-1, 1])
        mask = batch[5]
        target_item_seq = batch[6]
        target_cat_seq = batch[7]
        return hist_item_seq, hist_cat_seq, target_item, target_cat, label, mask, target_item_seq, target_cat_seq

    # define loss function by predicts and label
    def create_loss(self, raw_pred, label):
        avg_loss = paddle.nn.functional.binary_cross_entropy_with_logits(
            raw_pred, label, reduction='mean')
        return avg_loss

    # define optimizer
    def create_optimizer(self, dy_model, config):
        boundaries = [410000]
        base_lr = config.get(
            "hyper_parameters.optimizer.learning_rate_base_lr")
        values = [base_lr, 0.2]
        sgd_optimizer = paddle.optimizer.SGD(
            learning_rate=paddle.optimizer.lr.PiecewiseDecay(
                boundaries=boundaries, values=values),
            parameters=dy_model.parameters())
        return sgd_optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["auc"]
        #auc_metric = paddle.metric.Auc(num_thresholds=self.bucket)
        auc_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        hist_item_seq, hist_cat_seq, target_item, target_cat, label, mask, target_item_seq, target_cat_seq = self.create_feeds(
            batch_data, config)

        raw_pred = dy_model.forward(hist_item_seq, hist_cat_seq, target_item,
                                    target_cat, label, mask, target_item_seq,
                                    target_cat_seq)
        loss = self.create_loss(raw_pred, label)
        predict = paddle.nn.functional.sigmoid(raw_pred)
        predict_2d = paddle.concat([1 - predict, predict], 1)
        label_int = paddle.cast(label, 'int64')
        metrics_list[0].update(
            preds=predict_2d.numpy(), labels=label_int.numpy())

        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        hist_item_seq, hist_cat_seq, target_item, target_cat, label, mask, target_item_seq, target_cat_seq = self.create_feeds(
            batch_data, config)
        raw_pred = dy_model.forward(hist_item_seq, hist_cat_seq, target_item,
                                    target_cat, label, mask, target_item_seq,
                                    target_cat_seq)

        predict = paddle.nn.functional.sigmoid(raw_pred)
        predict_2d = paddle.concat([1 - predict, predict], 1)
        label_int = paddle.cast(label, 'int64')
        metrics_list[0].update(
            preds=predict_2d.numpy(), labels=label_int.numpy())

        return metrics_list, None
