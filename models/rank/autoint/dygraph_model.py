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
        feature_number = config.get("hyper_parameters.feature_number")
        embedding_dim = config.get("hyper_parameters.embedding_dim")
        fc_sizes = config.get("hyper_parameters.fc_sizes")
        use_residual = config.get("hyper_parameters.use_residual")
        scaling = config.get("hyper_parameters.scaling")
        use_wide = config.get("hyper_parameters.use_wide")
        use_sparse = config.get("hyper_parameters.use_sparse")
        head_num = config.get("hyper_parameters.head_num")
        num_field = config.get("hyper_parameters.num_field")
        attn_layer_sizes = config.get("hyper_parameters.attn_layer_sizes")

        autoint_model = net.AutoInt(feature_number, embedding_dim, fc_sizes, use_residual, scaling, use_wide, 
            use_sparse, head_num, num_field, attn_layer_sizes)
        return autoint_model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, batch_data, config):
        label = paddle.to_tensor(batch_data[0], dtype='int64')
        feat_index = paddle.to_tensor(batch_data[1], dtype='int64')
        feat_value = paddle.to_tensor(batch_data[2], dtype='float32')
        return label, feat_index, feat_value

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                label, dtype="float32"))
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
        metrics_list_name = ["auc"]
        auc_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        label, feat_index, feat_value = self.create_feeds(batch_data, config)

        pred = dy_model.forward(feat_index, feat_value)
        loss = self.create_loss(pred, label)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        print_dict = {'loss': loss}
        # print_dict = None
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        label, feat_index, feat_value = self.create_feeds(batch_data, config)

        pred = dy_model.forward(feat_index, feat_value)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        return metrics_list, None
