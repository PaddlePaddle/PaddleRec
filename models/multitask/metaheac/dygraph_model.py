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
import numpy as np
import pickle
import net


class DygraphModel():
    # define model
    def create_model(self, config):
        max_idxs = config.get("hyper_parameters.max_idxs")
        embed_dim = config.get("hyper_parameters.embed_dim")
        mlp_dims = config.get("hyper_parameters.mlp_dims")

        num_expert = config.get("hyper_parameters.num_expert")
        num_output = config.get("hyper_parameters.num_output")

        meta_model = net.WideAndDeepModel(max_idxs, embed_dim, mlp_dims, num_expert, num_output)
        # model_state_dict = paddle.load('paddle.pkl')
        # meta_model.set_dict(model_state_dict)

        return meta_model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, batch_data, config):
        x_spt =  batch_data[0]
        y_spt = batch_data[1]

        x_qry = batch_data[2]
        y_qry = batch_data[3]
        return x_spt, y_spt, x_qry, y_qry

    # define loss function by predicts and label
    def create_loss(self, pred, y_label):

        loss_ctr = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                y_label, dtype="float32"))
        return loss_ctr

    # define optimizer 
    def create_optimizer(self, dy_model, config, mode="train"):
        if mode == "train":
            lr = config.get("hyper_parameters.optimizer.global_learning_rate", 0.001)
            optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=dy_model.parameters())
        else:
            lr = config.get("hyper_parameters.optimizer.local_test_learning_rate", 0.001)
            optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["AUC"]
        auc_ctr_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_ctr_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self,  dy_model, metric_list, batch, config):
        # print(len(batch))
        # exit(0)
        x_spt, y_spt, x_qry, y_qry = self.create_feeds(batch, config)

        task_count = config.get("hyper_parameters.task_count",5)
        local_lr = config.get("hyper_parameters.local_lr",0.0002)
        criterion = paddle.nn.BCELoss()

        losses_q = []
        dy_model.clear_gradients()
        for i in range(task_count):
            ##  local update --------------
            fast_parameters = list(dy_model.parameters())
            for weight in fast_parameters:
                weight.fast = None

            support_set_y_pred = dy_model(x_spt[i])
            label = paddle.squeeze(y_spt[i].astype('float32'))

            loss = criterion(support_set_y_pred, label)
            dy_model.clear_gradients()
            loss.backward()

            fast_parameters = list(dy_model.parameters())
            for weight in fast_parameters:
                if weight.grad is None:
                    continue
                if weight.fast is None:
                    weight.fast = weight - local_lr * weight.grad  # create weight.fast
                else:
                    weight.fast = weight.fast - local_lr * weight.grad
            dy_model.clear_gradients()
            ##  local update --------------

            query_set_y_pred = dy_model(x_qry[i])
            label = paddle.squeeze(y_qry[i].astype('float32'))
            loss_q = criterion(query_set_y_pred, label)
            losses_q.append(loss_q)

        loss_average = paddle.stack(losses_q).mean(0)
        print_dict = {'loss': loss_average}

        return loss_average, metric_list, print_dict

    def infer_train_forward(self, dy_model, batch, config):
        batch_x, batch_y = batch[0], batch[1]
        criterion = paddle.nn.BCELoss()

        pred = dy_model.forward(batch_x)

        label = paddle.squeeze(batch_y.astype('float32'))
        loss_q = criterion(pred, label)

        return loss_q

    def infer_forward(self, dy_model, metric_list, metric_list_local, batch, config):
        batch_x, batch_y = batch[0], batch[1]
        pred = dy_model.forward(batch_x)
        label = paddle.squeeze(batch_y.astype('float32'))
        
        pred = paddle.unsqueeze(pred,1)
        pred = paddle.concat([1-pred,pred],1)

        metric_list[0].update(preds=pred.numpy(), labels=label.numpy())
        metric_list_local[0].update(preds=pred.numpy(), labels=label.numpy())

        return metric_list, metric_list_local
