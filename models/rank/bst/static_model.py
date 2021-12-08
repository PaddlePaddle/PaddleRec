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

import math
import paddle

from net import BSTLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.is_distributed = False
        self.distributed_embedding = False

        self.item_emb_size = self.config.get("hyper_parameters.item_emb_size",
                                             64)
        self.cat_emb_size = self.config.get("hyper_parameters.cat_emb_size",
                                            64)
        self.position_emb_size = self.config.get(
            "hyper_parameters.position_emb_size", 64)
        self.act = self.config.get("hyper_parameters.act", "sigmoid")
        self.is_sparse = self.config.get("hyper_parameters.is_sparse", False)
        # significant for speeding up the training process
        self.use_DataLoader = self.config.get(
            "hyper_parameters.use_DataLoader", False)
        self.item_count = self.config.get("hyper_parameters.item_count", 63001)
        self.user_count = self.config.get("hyper_parameters.user_count",
                                          192403)

        self.cat_count = self.config.get("hyper_parameters.cat_count", 801)
        self.position_count = self.config.get(
            "hyper_parameters.position_count", 5001)
        self.n_encoder_layers = self.config.get(
            "hyper_parameters.n_encoder_layers", 1)
        self.d_model = self.config.get("hyper_parameters.d_model", 96)
        self.d_key = self.config.get("hyper_parameters.d_key", None)
        self.d_value = self.config.get("hyper_parameters.d_value", None)
        self.n_head = self.config.get("hyper_parameters.n_head", None)
        self.dropout_rate = self.config.get("hyper_parameters.dropout_rate",
                                            0.0)
        self.postprocess_cmd = self.config.get(
            "hyper_parameters.postprocess_cmd", "da")
        self.preprocess_cmd = self.config.get(
            "hyper_parameters.postprocess_cmd", "n")
        self.prepostprocess_dropout = self.config.get(
            "hyper_parameters.prepostprocess_dropout", 0.0)
        self.d_inner_hid = self.config.get("hyper_parameters.d_inner_hid", 512)
        self.relu_dropout = self.config.get("hyper_parameters.relu_dropout",
                                            0.0)
        self.layer_sizes = self.config.get("hyper_parameters.fc_sizes", None)
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")

    def create_feeds(self, is_infer=False):

        userid = paddle.static.data(
            name="userid", shape=[None, 1], dtype="int64")

        history = paddle.static.data(
            name="history", shape=[None, -1], dtype="int64")

        cate = paddle.static.data(name="cate", shape=[None, -1], dtype="int64")

        position = paddle.static.data(
            name="position", shape=[None, -1], dtype="int64")

        target = paddle.static.data(
            name="target", shape=[None, 1], dtype="int64")

        target_cate = paddle.static.data(
            name="target_cate", shape=[None, 1], dtype="int64")

        target_position = paddle.static.data(
            name="target_position", shape=[None, 1], dtype="int64")

        label = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64")

        feeds_list = [label] + [userid] + [history] + [cate] + [position] + [
            target
        ] + [target_cate] + [target_position]
        return feeds_list

    def net(self, input, is_infer=False):
        self.label_input = input[0]
        self.user_input = input[1]
        self.hist_input = input[2]
        self.cate_input = input[3]
        self.pos_input = input[4]
        self.target_input = input[5]
        self.target_cate_input = input[6]
        self.target_pos_input = input[7]

        bst_model = BSTLayer(
            self.user_count, self.item_emb_size, self.cat_emb_size,
            self.position_emb_size, self.act, self.is_sparse,
            self.use_DataLoader, self.item_count, self.cat_count,
            self.position_count, self.n_encoder_layers, self.d_model,
            self.d_key, self.d_value, self.n_head, self.dropout_rate,
            self.postprocess_cmd, self.preprocess_cmd,
            self.prepostprocess_dropout, self.d_inner_hid, self.relu_dropout,
            self.layer_sizes)

        pred = bst_model.forward(
            self.user_input, self.hist_input, self.cate_input, self.pos_input,
            self.target_input, self.target_cate_input, self.target_pos_input)

        #pred = F.sigmoid(prediction)

        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)

        auc, batch_auc_var, _ = paddle.static.auc(input=predict_2d,
                                                  label=self.label_input,
                                                  slide_steps=0)

        self.inference_target_var = auc
        if is_infer:
            fetch_dict = {'auc': auc}
            return fetch_dict

        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                self.label_input, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        self._cost = avg_cost
        fetch_dict = {'cost': avg_cost, 'auc': auc}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[10, 20, 50],
            values=[0.001, 0.0001, 0.0005, 0.00001],
            verbose=True)
        optimizer = paddle.optimizer.Adagrad(learning_rate=scheduler)
        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)
