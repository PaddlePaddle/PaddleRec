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
        user_size = config.get("hyper_parameters.user_size")
        cms_segid_size = config.get("hyper_parameters.cms_segid_size")
        cms_group_id_size = config.get("hyper_parameters.cms_group_id_size")
        final_gender_code_size = config.get(
            "hyper_parameters.final_gender_code_size")
        age_level_size = config.get("hyper_parameters.age_level_size")
        pvalue_level_size = config.get("hyper_parameters.pvalue_level_size")
        shopping_level_size = config.get(
            "hyper_parameters.shopping_level_size")
        occupation_size = config.get("hyper_parameters.occupation_size")
        new_user_class_level_size = config.get(
            "hyper_parameters.new_user_class_level_size")
        adgroup_id_size = config.get("hyper_parameters.adgroup_id_size")
        cate_size = config.get("hyper_parameters.cate_size")
        campaign_id_size = config.get("hyper_parameters.campaign_id_size")
        customer_size = config.get("hyper_parameters.customer_size")
        brand_size = config.get("hyper_parameters.brand_size")
        btag_size = config.get("hyper_parameters.btag_size")
        pid_size = config.get("hyper_parameters.pid_size")
        main_embedding_size = config.get(
            "hyper_parameters.main_embedding_size")
        other_embedding_size = config.get(
            "hyper_parameters.other_embedding_size")

        dmr_model = net.DMRLayer(
            user_size, cms_segid_size, cms_group_id_size,
            final_gender_code_size, age_level_size, pvalue_level_size,
            shopping_level_size, occupation_size, new_user_class_level_size,
            adgroup_id_size, cate_size, campaign_id_size, customer_size,
            brand_size, btag_size, pid_size, main_embedding_size,
            other_embedding_size)

        return dmr_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        b = batch_data[0]
        sparse_tensor = b.astype('int64')
        dense_tensor = paddle.to_tensor(b[:, 264].numpy().astype('float32')
                                        .reshape(-1, 1))
        label = sparse_tensor[:, -1].reshape([-1, 1])
        return label, [sparse_tensor, dense_tensor]

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

        pred, loss = dy_model.forward(input_tensor, False)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        print_dict = {'loss': loss}
        # print_dict = None
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        label, input_tensor = self.create_feeds(batch_data, config)

        pred, loss = dy_model.forward(input_tensor, True)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        return metrics_list, None
