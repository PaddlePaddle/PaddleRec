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
from metrics import LogLoss

import net


class DygraphModel():
    # define model
    def create_model(self, config):
        stage = int(config.get('stage', 0))
        num_inputs = config.get("hyper_parameters.num_inputs")
        input_size = config.get("hyper_parameters.input_size")
        embedding_size = config.get("hyper_parameters.embedding_size")
        width = config.get("hyper_parameters.width")
        depth = config.get('hyper_parameters.depth')
        pairs = config.get('hyper_parameters.pairs')
        deepfm_model = net.AutoDeepFMLayer(
            num_inputs, input_size, embedding_size, width, depth, pairs, stage)

        return deepfm_model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, batch_data, config):
        return batch_data

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        loss = F.binary_cross_entropy(pred, label.astype('float32'))
        return loss

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        if int(config.get('stage', 0)) == 1:
            optimizer1 = paddle.optimizer.Adam(
                learning_rate=lr, parameters=dy_model.parameters())
            optimizer2 = None
        else:
            from optimizer import SimpleGrda
            params = [p for n, p in dy_model.named_parameters() if n != 'mask']
            mask_params = [
                p for n, p in dy_model.named_parameters() if n == 'mask'
            ]
            optimizer1 = paddle.optimizer.Adam(
                learning_rate=lr, parameters=params)
            optimizer2 = SimpleGrda(mask_params, 1,
                                    config['hyper_parameters.grad_c'],
                                    config['hyper_parameters.grad_mu'])

        return optimizer1, optimizer2

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["auc", "log_loss"]
        auc_metric = paddle.metric.Auc("ROC")
        metrics_list = [auc_metric, LogLoss()]
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        inputs, label = self.create_feeds(batch_data, config)

        pred = dy_model.forward(inputs)
        loss = self.create_loss(pred, label)
        # update metrics
        pred = pred.unsqueeze(-1)
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        inputs, label = self.create_feeds(batch_data, config)
        pred = dy_model.forward(inputs)
        pred = pred.unsqueeze(-1)
        metrics_list[1].update(pred, label.astype('float32').unsqueeze(-1))

        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        return metrics_list, None
