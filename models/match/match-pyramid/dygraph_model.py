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
        emb_path = config.get("hyper_parameters.emb_path")
        vocab_size = config.get("hyper_parameters.vocab_size")
        emb_size = config.get("hyper_parameters.emb_size")
        kernel_num = config.get("hyper_parameters.kernel_num")
        conv_filter = config.get("hyper_parameters.conv_filter")
        conv_act = config.get("hyper_parameters.conv_act")
        hidden_size = config.get("hyper_parameters.hidden_size")
        out_size = config.get("hyper_parameters.out_size")
        pool_size = config.get("hyper_parameters.pool_size")
        pool_stride = config.get("hyper_parameters.pool_stride")
        pool_padding = config.get("hyper_parameters.pool_padding")
        pool_type = config.get("hyper_parameters.pool_type")
        hidden_act = config.get("hyper_parameters.hidden_act")

        pyramid_model = net.MatchPyramidLayer(
            emb_path, vocab_size, emb_size, kernel_num, conv_filter, conv_act,
            hidden_size, out_size, pool_size, pool_stride, pool_padding,
            pool_type, hidden_act)
        return pyramid_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, sentence_left_size,
                     sentence_right_size):
        sentence_left = paddle.to_tensor(batch_data[0].numpy().astype('int64')
                                         .reshape(-1, sentence_left_size))
        sentence_right = paddle.to_tensor(batch_data[1].numpy().astype('int64')
                                          .reshape(-1, sentence_right_size))
        return [sentence_left, sentence_right]

    # define loss function by predicts and label
    def create_loss(self, prediction):
        pos = paddle.slice(
            prediction, axes=[0, 1], starts=[0, 0], ends=[64, 1])
        neg = paddle.slice(
            prediction, axes=[0, 1], starts=[64, 0], ends=[128, 1])
        loss_part1 = paddle.subtract(
            paddle.full(
                shape=[64, 1], fill_value=1.0, dtype='float32'), pos)
        loss_part2 = paddle.add(loss_part1, neg)
        loss_part3 = paddle.maximum(
            paddle.full(
                shape=[64, 1], fill_value=0.0, dtype='float32'),
            loss_part2)
        avg_cost = paddle.mean(loss_part3)
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
        metrics_list_name = []
        metrics_list = []
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        sentence_left_size = config.get("hyper_parameters.sentence_left_size")
        sentence_right_size = config.get(
            "hyper_parameters.sentence_right_size")
        batch_size = config.get("runner.train_batch_size", 128)
        inputs = self.create_feeds(batch_data, sentence_left_size,
                                   sentence_right_size)

        prediction = dy_model.forward(inputs)
        loss = self.create_loss(prediction)
        # update metrics
        print_dict = {"loss": loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        sentence_left_size = config.get("hyper_parameters.sentence_left_size")
        sentence_right_size = config.get(
            "hyper_parameters.sentence_right_size")
        batch_size = config.get("runner.infer_batch_size", 128)
        inputs = self.create_feeds(batch_data, sentence_left_size,
                                   sentence_right_size)

        prediction = dy_model.forward(inputs)
        # update metrics
        print_dict = {"prediction": prediction}
        return metrics_list, print_dict
