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

import net
import sys


class DygraphModel:
    def create_model(self, config):
        article_content_size = config.get("hyper_parameters.article_content_size")
        article_title_size = config.get("hyper_parameters.article_title_size")
        browse_size = config.get("hyper_parameters.browse_size")
        neg_condidate_sample_size = config.get(
            "hyper_parameters.neg_condidate_sample_size"
        )
        word_dimension = config.get("hyper_parameters.word_dimension")
        category_size = config.get("hyper_parameters.category_size")
        sub_category_size = config.get("hyper_parameters.sub_category_size")
        cate_dimension = config.get("hyper_parameters.category_dimension")
        word_dict_size = config.get("hyper_parameters.word_dict_size")

        # Fastformer
        hidden_size = config.get("hyper_parameters.hidden_size")
        hidden_dropout_prob = config.get("hyper_parameters.hidden_dropout_prob")
        num_hidden_layers = config.get("hyper_parameters.num_hidden_layers")
        num_attention_heads = config.get("hyper_parameters.num_attention_heads")
        intermediate_size = config.get("hyper_parameters.intermediate_size")
        max_position_embeddings = config.get("hyper_parameters.max_position_embeddings")
        # vocab_size = config.get("hyper_parameters.vocab_size")
        layer_norm_eps = float(config.get("hyper_parameters.layer_norm_eps"))
        initializer_range = config.get("hyper_parameters.initializer_range")
        pooler_type = config.get("hyper_parameters.pooler_type")
        # enable_fp16 = config.get("hyper_parameters.enable_fp16")

        return net.FastRecommender(
            article_content_size,
            article_title_size,
            browse_size,
            neg_condidate_sample_size,
            word_dimension,
            category_size,
            sub_category_size,
            cate_dimension,
            word_dict_size,
            hidden_size,
            num_attention_heads,
            initializer_range,
            layer_norm_eps,
            intermediate_size,
            hidden_dropout_prob,
            num_hidden_layers,
            max_position_embeddings,
            pooler_type,
        )

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch, config):
        label = batch[0]
        return label, batch[1:]

    # define loss function by predicts and label
    def create_loss(self, raw_pred, label):
        cost = paddle.nn.functional.cross_entropy(
            input=raw_pred, label=paddle.cast(label, "float32"), soft_label=True
        )
        avg_cost = paddle.mean(x=cost)
        return avg_cost

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=dy_model.parameters()
        )
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        # metrics_list_name = ["acc"]
        # auc_metric = paddle.metric.Accuracy()
        metrics_list_name = ["auc"]
        auc_metric = paddle.metric.Auc()
        metrics_list = [auc_metric]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        labels, sparse_tensor = self.create_feeds(batch_data, config)

        raw = dy_model.forward(sparse_tensor)

        loss = paddle.nn.functional.cross_entropy(
            input=raw, label=paddle.cast(labels, "float32"), soft_label=True
        )

        soft_predict = paddle.nn.functional.sigmoid(paddle.reshape(raw, [-1, 1]))
        predict_2d = paddle.concat(x=[1 - soft_predict, soft_predict], axis=-1)
        labels = paddle.reshape(labels, [-1, 1])
        metrics_list[0].update(preds=predict_2d.numpy(), labels=labels.numpy())

        loss = paddle.mean(loss)
        print_dict = None
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        labels, sparse_tensor = self.create_feeds(batch_data, config)
        raw = dy_model.forward(sparse_tensor)
        # predict_raw = paddle.nn.functional.softmax(raw)

        soft_predict = paddle.nn.functional.sigmoid(paddle.reshape(raw, [-1, 1]))
        predict_2d = paddle.concat(x=[1 - soft_predict, soft_predict], axis=-1)
        labels = paddle.reshape(labels, [-1, 1])
        metrics_list[0].update(preds=predict_2d.numpy(), labels=labels.numpy())
        return metrics_list, None
