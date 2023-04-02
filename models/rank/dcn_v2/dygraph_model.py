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
        is_Stacked = config.get("hyper_parameters.is_Stacked", None)
        use_low_rank_mixture = config.get(
            "hyper_parameters.use_low_rank_mixture", None)
        low_rank = config.get("hyper_parameters.low_rank", 32)
        num_experts = config.get("hyper_parameters.num_experts", 4)
        dnn_use_bn = config.get("hyper_parameters.dnn_use_bn", None)

        dcn_v2_model = net.DCN_V2Layer(
            sparse_feature_number, sparse_feature_dim, dense_feature_dim,
            sparse_input_slot - 1, fc_sizes, cross_num, is_Stacked,
            use_low_rank_mixture, low_rank, num_experts)

        return dcn_v2_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        # print("----batch_data", batch_data[0])
        # print("----batch_data", batch_data[1])
        # print("----batch_data", batch_data[-1])
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
        # print("-----dygraph-----dense_tensor:----", dense_tensor)
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
        clip_by_norm = config.get("hyper_parameters.optimizer.clip_by_norm",
                                  10.0)
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_by_norm)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=dy_model.parameters(), grad_clip=clip)
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
        pred = dy_model.forward(sparse_tensor, dense_tensor)

        log_loss = self.create_loss(pred, label)

        # l2_reg_cross = config.get("hyper_parameters.l2_reg_cross", None)

        # for param in dy_model.DeepCrossLayer_.W.parameters():
        #     log_loss += l2_reg_cross * paddle.norm(param, p=2)

        # loss = log_loss + l2_loss
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        # print("---dygraph----pred,loss,predict_2d---",pred,loss,predict_2d)
        # print("---dygraph----metrics_list",metrics_list)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        # print_dict format :{'loss': loss}
        print_dict = {'log_loss': log_loss}
        # print_dict = None
        return log_loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        label, sparse_tensor, dense_tensor = self.create_feeds(batch_data,
                                                               config)
        # print("----label, sparse_tensor, dense_tensor",label, sparse_tensor, dense_tensor)
        pred = dy_model.forward(sparse_tensor, dense_tensor)
        # update metrics
        log_loss = self.create_loss(pred, label)
        print_dict = {'log_loss': log_loss}

        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        # print("---pred,predict_2d---",pred,predict_2d)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        # print("---metrics_list",metrics_list)
        return metrics_list, print_dict
        # return metrics_list, None
