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
        max_len = config.get("hyper_parameters.max_len", 3)
        sparse_feature_number = config.get(
            "hyper_parameters.sparse_feature_number")
        sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
        num_field = config.get("hyper_parameters.num_field")
        learning_rate = config.get("hyper_parameters.optimizer.learning_rate")
        ctr_fc_sizes = config.get("hyper_parameters.ctr_fc_sizes")
        cvr_fc_sizes = config.get("hyper_parameters.cvr_fc_sizes")
        sparse_feature_number = config.get(
            "hyper_parameters.sparse_feature_number")

        esmm_model = net.ESMMLayer(sparse_feature_number, sparse_feature_dim,
                                   num_field, ctr_fc_sizes, cvr_fc_sizes)

        return esmm_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        max_len = config.get("hyper_parameters.max_len", 3)
        sparse_tensor = []
        for b in batch_data[:-2]:
            sparse_tensor.append(
                paddle.to_tensor(b.numpy().astype('int64').reshape(-1,
                                                                   max_len)))
        ctr_label = paddle.to_tensor(batch_data[-2].numpy().astype('int64')
                                     .reshape(-1, 1))
        ctcvr_label = paddle.to_tensor(batch_data[-1].numpy().astype('int64')
                                       .reshape(-1, 1))
        return sparse_tensor, ctr_label, ctcvr_label

    # define loss function by predicts and label
    def create_loss(self, ctr_out_one, ctr_clk, ctcvr_prop_one, ctcvr_buy):
        loss_ctr = paddle.nn.functional.log_loss(
            input=ctr_out_one, label=paddle.cast(
                ctr_clk, dtype="float32"))
        loss_ctcvr = paddle.nn.functional.log_loss(
            input=ctcvr_prop_one,
            label=paddle.cast(
                ctcvr_buy, dtype="float32"))
        cost = loss_ctr + loss_ctcvr
        avg_cost = paddle.mean(x=cost)
        return avg_cost

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["auc_ctr", "auc_ctcvr"]
        auc_ctr_metric = paddle.metric.Auc("ROC")
        auc_ctcvr_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_ctr_metric, auc_ctcvr_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        sparse_tensor, label_ctr, label_ctcvr = self.create_feeds(batch_data,
                                                                  config)

        ctr_out, ctr_out_one, cvr_out, cvr_out_one, ctcvr_prop, ctcvr_prop_one = dy_model.forward(
            sparse_tensor)
        loss = self.create_loss(ctr_out_one, label_ctr, ctcvr_prop_one,
                                label_ctcvr)
        # update metrics
        metrics_list[0].update(preds=ctr_out.numpy(), labels=label_ctr.numpy())
        metrics_list[1].update(
            preds=ctcvr_prop.numpy(), labels=label_ctcvr.numpy())

        # print_dict format :{'loss': loss}
        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        sparse_tensor, label_ctr, label_ctcvr = self.create_feeds(batch_data,
                                                                  config)

        ctr_out, ctr_out_one, cvr_out, cvr_out_one, ctcvr_prop, ctcvr_prop_one = dy_model.forward(
            sparse_tensor)
        # update metrics
        metrics_list[0].update(preds=ctr_out.numpy(), labels=label_ctr.numpy())
        metrics_list[1].update(
            preds=ctcvr_prop.numpy(), labels=label_ctcvr.numpy())

        return metrics_list, None
