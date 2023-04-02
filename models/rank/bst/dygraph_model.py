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
        item_emb_size = config.get("hyper_parameters.item_emb_size", 64)
        cat_emb_size = config.get("hyper_parameters.cat_emb_size", 64)
        position_emb_size = config.get("hyper_parameters.position_emb_size",
                                       64)
        act = config.get("hyper_parameters.act", "sigmoid")
        is_sparse = config.get("hyper_parameters.is_sparse", False)
        # significant for speeding up the training process
        use_DataLoader = config.get("hyper_parameters.use_DataLoader", False)
        item_count = config.get("hyper_parameters.item_count", 63001)
        user_count = config.get("hyper_parameters.user_count", 192403)

        cat_count = config.get("hyper_parameters.cat_count", 801)
        position_count = config.get("hyper_parameters.position_count", 5001)
        n_encoder_layers = config.get("hyper_parameters.n_encoder_layers", 1)
        d_model = config.get("hyper_parameters.d_model", 96)
        d_key = config.get("hyper_parameters.d_key", None)
        d_value = config.get("hyper_parameters.d_value", None)
        n_head = config.get("hyper_parameters.n_head", None)
        dropout_rate = config.get("hyper_parameters.dropout_rate", 0.0)
        postprocess_cmd = config.get("hyper_parameters.postprocess_cmd", "da")
        preprocess_cmd = config.get("hyper_parameters.postprocess_cmd", "n")
        prepostprocess_dropout = config.get(
            "hyper_parameters.prepostprocess_dropout", 0.0)
        d_inner_hid = config.get("hyper_parameters.d_inner_hid", 512)
        relu_dropout = config.get("hyper_parameters.relu_dropout", 0.0)
        layer_sizes = config.get("hyper_parameters.fc_sizes", None)

        bst_model = net.BSTLayer(
            user_count, item_emb_size, cat_emb_size, position_emb_size, act,
            is_sparse, use_DataLoader, item_count, cat_count, position_count,
            n_encoder_layers, d_model, d_key, d_value, n_head, dropout_rate,
            postprocess_cmd, preprocess_cmd, prepostprocess_dropout,
            d_inner_hid, relu_dropout, layer_sizes)

        return bst_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
        sparse_tensor = []
        for b in batch_data:
            sparse_tensor.append(
                paddle.to_tensor(b.numpy().astype('int64').reshape(-1,
                                                                   len(b[0]))))
        label = sparse_tensor[0]
        return label, sparse_tensor[1], sparse_tensor[2], sparse_tensor[
            3], sparse_tensor[4], sparse_tensor[5], sparse_tensor[
                6], sparse_tensor[7]

    # define loss function by predicts and label
    def create_loss(self, pred, label):
        cost = paddle.nn.functional.log_loss(
            pred, label=paddle.cast(
                label, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        return avg_cost

    # define optimizer
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[10, 20, 50],
            values=[0.001, 0.0001, 0.0005, 0.00001],
            verbose=True)
        optimizer = paddle.optimizer.Adagrad(
            learning_rate=scheduler, parameters=dy_model.parameters())
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
        label, userid, history, cate, position, target, target_cate, target_position = self.create_feeds(
            batch_data, config)
        pred = dy_model.forward(userid, history, cate, position, target,
                                target_cate, target_position)
        loss = self.create_loss(pred, label)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())

        # print_dict format :{'loss': loss}
        print_dict = None
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        label, userid, history, cate, position, target, target_cate, target_position = self.create_feeds(
            batch_data, config)

        pred = dy_model.forward(userid, history, cate, position, target,
                                target_cate, target_position)
        # update metrics
        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metrics_list[0].update(preds=predict_2d.numpy(), labels=label.numpy())
        return metrics_list, None
