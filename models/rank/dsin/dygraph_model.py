# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
        user_size = config.get("hyper_parameters.user_size")
        cms_segid_size = config.get("hyper_parameters.cms_segid_size")
        cms_group_size = config.get("hyper_parameters.cms_group_size")
        final_gender_size = config.get("hyper_parameters.final_gender_size")
        age_level_size = config.get("hyper_parameters.age_level_size")
        pvalue_level_size = config.get("hyper_parameters.pvalue_level_size")
        shopping_level_size = config.get(
            "hyper_parameters.shopping_level_size")
        occupation_size = config.get("hyper_parameters.occupation_size")
        new_user_class_level_size = config.get(
            "hyper_parameters.new_user_class_level_size")
        adgroup_size = config.get("hyper_parameters.adgroup_size")
        cate_size = config.get("hyper_parameters.cate_size")
        campaign_size = config.get("hyper_parameters.campaign_size")
        customer_size = config.get("hyper_parameters.customer_size")
        brand_size = config.get("hyper_parameters.brand_size")
        pid_size = config.get("hyper_parameters.pid_size")
        feat_embed_size = config.get("hyper_parameters.feat_embed_size")

        dsin_model = net.DSIN_layer(
            user_size,
            adgroup_size,
            pid_size,
            cms_segid_size,
            cms_group_size,
            final_gender_size,
            age_level_size,
            pvalue_level_size,
            shopping_level_size,
            occupation_size,
            new_user_class_level_size,
            campaign_size,
            customer_size,
            cate_size,
            brand_size,
            sparse_embed_size=feat_embed_size,
            l2_reg_embedding=1e-6)

        return dsin_model

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        return paddle.nn.BCELoss()(pred, label)

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        data, label = (batch_data[0], batch_data[1], batch_data[2],
                       batch_data[3]), batch_data[-1]
        #data, label = batch_data[0], batch_data[1]
        label = label.reshape([-1, 1])
        return label, data

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
        label, input_tensor = self.create_feeds(batch_data, config)

        pred = dy_model.forward(input_tensor)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        loss = self.create_loss(pred, paddle.cast(label, "float32"))
        print_dict = {'loss': loss}
        # print_dict = None
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        label, input_tensor = self.create_feeds(batch_data, config)

        pred = dy_model.forward(input_tensor)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        return metrics_list, None
