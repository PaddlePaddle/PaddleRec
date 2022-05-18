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
        # max_sentence = config.get('hyper_parameters.max_sentence', 50)
        # max_all = config.get('hyper_parameters.max_all', 50)
        # max_sent_length = config.get('hyper_parameters.max_sent_length', 50)
        # max_sents = config.get('hyper_parameters.max_sents', 50)
        # max_entity_num = config.get('hyper_parameters.max_entity_num', 50)
        # num = config.get('hyper_parameters.num', 100)
        # num1 = config.get('hyper_parameters.num1', 100)
        # num2 = config.get('hyper_parameters.num2', 100)
        # npratio = config.get('hyper_parameters.npratio', 0.5)
        # hidden_size = config.get('hyper_parameters.hidden_size', 100)
        embedding_size = config.get('hyper_parameters.embedding_size', 100)
        vocab_size = config.get('hyper_parameters.vocab_size', 100)
        kim_model = net.KIMLayer(vocab_size, embedding_size)
        return kim_model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, batch_data):
        return [x.squeeze(0) for x in batch_data]

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        return F.cross_entropy(pred, label)

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.00005)
        optimizer = paddle.optimizer.Adagrad(
            learning_rate=lr,
            initial_accumulator_value=1e-8,
            parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric

    def create_metrics(self):
        metrics_list_name = ["Acc"]
        metrics_list = [paddle.metric.Accuracy()]
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        *inputs, labels = self.create_feeds(batch_data)
        labels = labels.argmax(-1,keepdim=True)
        prediction = dy_model.forward(*inputs)
        loss = self.create_loss(prediction, labels)
        # update metrics
        print_dict = {"loss": loss}
        correct = metrics_list[0].compute(prediction, labels)
        metrics_list[0].update(correct)
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        inputs = self.create_feeds(batch_data)

        prediction = dy_model.forward(*inputs)
        # update metrics
        print_dict = {
            "y_pred": F.softmax(prediction, -1),
        }
        return metrics_list, print_dict
