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
    def create_model(self, config):
        feature_vocabulary = config.get("hyper_parameters.feature_vocabulary")
        embedding_size = config.get("hyper_parameters.embedding_size")
        tower_dims = config.get("hyper_parameters.dims")
        drop_prob = config.get('hyper_parameters.drop_prob')
        feature_vocabulary = dict(feature_vocabulary)
        model = net.AITM(
            feature_vocabulary,
            embedding_size,
            tower_dims,
            drop_prob
        )
        return model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, batch_data, config):
        click, conversion, features = batch_data
        return click.astype('float32'), conversion.astype('float32'), features

    # define loss function by predicts and label
    def create_loss(self, click_pred, conversion_pred, click_label, conversion_label, constraint_weight=0.6):
        click_loss = F.binary_cross_entropy(click_pred, click_label)
        conversion_loss = F.binary_cross_entropy(conversion_pred, conversion_label)

        label_constraint = paddle.maximum(conversion_pred - click_pred,
                                          paddle.zeros_like(click_label))
        constraint_loss = paddle.sum(label_constraint)

        loss = click_loss + conversion_loss + constraint_weight * constraint_loss
        return loss

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.0001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr,
            parameters=dy_model.parameters(),
            weight_decay=1e-6
        )
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["click_auc", "purchase_auc"]
        metrics_list = [paddle.metric.Auc("ROC", num_thresholds=100000), paddle.metric.Auc("ROC", num_thresholds=100000)]
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        click, conversion, features = self.create_feeds(batch_data, config)
        click_pred, conversion_pred = dy_model.forward(features)
        loss = self.create_loss(click_pred, conversion_pred, click, conversion)
        # update metrics

        self.update_auc(click_pred, click, metrics_list[0])
        self.update_auc(conversion_pred, conversion, metrics_list[1])
        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict

    @staticmethod
    def update_auc(prob, label, metrics):
        if prob.ndim == 1:
            prob = prob.unsqueeze(-1)
        assert prob.ndim == 2
        predict_2d = paddle.concat(x=[1 - prob, prob], axis=1)
        metrics.update(predict_2d, label)

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        click, conversion, features = self.create_feeds(batch_data, config)
        with paddle.no_grad():
            click_pred, conversion_pred = dy_model.forward(features)
        # update metrics
        self.update_auc(click_pred, click, metrics_list[0])
        self.update_auc(conversion_pred, conversion, metrics_list[1])
        return metrics_list, None

    def forward(self, dy_model, batch_data, config):
        click, conversion, features = self.create_feeds(batch_data, config)
        with paddle.no_grad():
            click_pred, conversion_pred = dy_model.forward(features)
        # update metrics
        return click, click_pred, conversion, conversion_pred
