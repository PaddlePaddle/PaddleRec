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
        sparse_feature_number = config.get(
            "hyper_parameters.sparse_feature_number")
        sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")

        fc_sizes = config.get("hyper_parameters.fc_sizes")
        dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
        sparse_input_slot = config.get('hyper_parameters.sparse_inputs_slots')
        cross_num = config.get("hyper_parameters.cross_num")
        l2_reg_cross = config.get("hyper_parameters.l2_reg_cross", None)
        clip_by_norm = config.get("hyper_parameters.clip_by_norm", None)
        is_sparse = config.get("hyper_parameters.is_sparse", None)
        dnn_use_bn = config.get("hyper_parameters.dnn_use_bn", None)
        deepcro_model = net.DeepCroLayer(
            sparse_feature_number, sparse_feature_dim, dense_feature_dim,
            sparse_input_slot - 1, fc_sizes, cross_num, clip_by_norm,
            l2_reg_cross, is_sparse
        )  # print("----dygraphModel---deepcro_model--", deepcro_model)
        return deepcro_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        # print("----batch_data", batch_data)
        dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
        sparse_tensor = []
        for b in batch_data[:-1]:
            sparse_tensor.append(
                paddle.to_tensor(b.numpy().astype('int64').reshape(-1, 1)))
        dense_tensor = paddle.to_tensor(batch_data[-1].numpy().astype(
            'float32').reshape(-1, dense_feature_dim))
        label = sparse_tensor[0]
        # print("-----dygraph-----label:----",label.shape)
        # print("-----dygraph-----sparse_tensor[1:]:----", sparse_tensor[1:])
        # print("-----dygraph-----dense_tensor:----", dense_tensor.shape)
        return label, sparse_tensor[1:], dense_tensor

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        # print("---dygraph----pred, label:",pred, label)
        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                label, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        # add l2_loss.............
        # print("---dygraph-----cost,avg_cost----",cost,avg_cost)
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
        label, sparse_tensor, dense_tensor = self.create_feeds(batch_data,
                                                               config)
        # print("---dygraph-----label, sparse_tensor, dense_tensor",label, sparse_tensor, dense_tensor)
        pred, l2_loss = dy_model.forward(sparse_tensor, dense_tensor)
        loss = self.create_loss(pred, label) + l2_loss
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        # print("---dygraph----pred,loss,predict_2d---",pred,loss,predict_2d)
        # print("---dygraph----metrics_list",metrics_list)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        # print_dict format :{'loss': loss}
        print_dict = None
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        label, sparse_tensor, dense_tensor = self.create_feeds(batch_data,
                                                               config)
        # print("----label, sparse_tensor, dense_tensor",label, sparse_tensor, dense_tensor)
        pred, l2_loss = dy_model.forward(sparse_tensor, dense_tensor)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        # print("---pred,predict_2d---",pred,predict_2d)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        # print("---metrics_list",metrics_list)
        return metrics_list, None
