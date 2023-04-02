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
import copy
import numpy as np
import net


class DygraphModel():
    # define model
    def create_model(self, config):
        conv_stride = config.get("hyper_parameters.conv_stride")
        conv_padding = config.get("hyper_parameters.conv_padding")
        conv_kernal = config.get("hyper_parameters.conv_kernal")
        bn_channel = config.get("hyper_parameters.bn_channel")
        maml_model = net.MAMLLayer(conv_stride, conv_padding, conv_kernal,
                                   bn_channel)
        return maml_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        x_spt = paddle.to_tensor(batch_data[0].numpy().astype("float32"))
        y_spt = paddle.to_tensor(batch_data[1].numpy().astype("int64"))
        x_qry = paddle.to_tensor(batch_data[2].numpy().astype("float32"))
        y_qry = paddle.to_tensor(batch_data[3].numpy().astype("int64"))
        #print("x_spt",x_spt.shape,"y_spt",y_spt.shape,"x_qry",x_qry.shape,"y_qry",y_qry.shape)
        return x_spt, y_spt, x_qry, y_qry

    # define optimizer
    def create_optimizer(self, dy_model, config):
        meta_lr = config.get("hyper_parameters.meta_optimizer.learning_rate",
                             0.001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=meta_lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = []
        metrics_list = []
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        np.random.seed(12345)
        x_spt, y_spt, x_qry, y_qry = self.create_feeds(batch_data, config)
        update_step = config.get("hyper_parameters.update_step", 5)
        task_num = x_spt.shape[0]
        query_size = x_qry.shape[
            1]  # 75 = 15 * 5, x_qry.shape = [32,75,1,28,28]
        loss_list = []
        loss_list.clear()
        correct_list = []
        correct_list.clear()
        task_grad = [[] for _ in range(task_num)]

        for i in range(task_num):
            # 外循环
            task_net = copy.deepcopy(dy_model)
            base_lr = config.get(
                "hyper_parameters.base_optimizer.learning_rate", 0.1)
            task_optimizer = paddle.optimizer.SGD(
                learning_rate=base_lr, parameters=task_net.parameters())
            for j in range(update_step):
                #内循环
                task_optimizer.clear_grad()  # 梯度清零
                y_hat = task_net.forward(x_spt[i])  # (setsz, ways) [5,5]
                loss_spt = F.cross_entropy(y_hat, y_spt[i])
                loss_spt.backward()
                task_optimizer.step()

            y_hat = task_net.forward(x_qry[i])
            loss_qry = F.cross_entropy(y_hat, y_qry[i])
            loss_qry.backward()
            for k in task_net.parameters():
                task_grad[i].append(k.grad)
            loss_list.append(loss_qry)
            pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)
            correct = paddle.equal(pred_qry, y_qry[i]).numpy().sum().item()
            correct_list.append(correct)

        loss_average = paddle.add_n(loss_list) / task_num
        acc = sum(correct_list) / (query_size * task_num)

        for num, k in enumerate(dy_model.parameters()):
            tmp_list = [task_grad[i][num] for i in range(task_num)]
            if tmp_list[0] is not None:
                k._set_grad_ivar(paddle.add_n(tmp_list) / task_num)

        acc = paddle.to_tensor(acc)
        print_dict = {'loss': loss_average, "acc": acc}
        _ = paddle.ones(shape=[5, 5], dtype="float32")
        return _, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        dy_model.train()
        x_spt, y_spt, x_qry, y_qry = self.create_feeds(batch_data, config)
        x_spt = x_spt[0]
        y_spt = y_spt[0]
        x_qry = x_qry[0]
        y_qry = y_qry[0]
        update_step = config.get("hyper_parameters.update_step_test", 5)
        query_size = x_qry.shape[0]
        correct_list = []
        correct_list.clear()

        task_net = copy.deepcopy(dy_model)
        base_lr = config.get("hyper_parameters.base_optimizer.learning_rate",
                             0.1)
        task_optimizer = paddle.optimizer.SGD(learning_rate=base_lr,
                                              parameters=task_net.parameters())
        for j in range(update_step):
            task_optimizer.clear_grad()
            y_hat = task_net.forward(x_spt)
            loss_spt = F.cross_entropy(y_hat, y_spt)
            loss_spt.backward()
            task_optimizer.step()

        y_hat = task_net.forward(x_qry)
        pred_qry = F.softmax(y_hat, axis=1).argmax(axis=1)
        correct = paddle.equal(pred_qry, y_qry).numpy().sum().item()
        correct_list.append(correct)
        acc = sum(correct_list) / query_size
        acc = paddle.to_tensor(acc)
        print_dict = {"acc": acc}

        return metrics_list, print_dict
