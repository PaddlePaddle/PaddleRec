# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import paddle

from net import DINLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.infer_target_var = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.is_distributed = False
        self.distributed_embedding = False
        if self.config.get("hyper_parameters.distributed_embedding", 0) == 1:
            self.distributed_embedding = True

        self.item_emb_size = self.config.get("hyper_parameters.item_emb_size",
                                             64)
        self.cat_emb_size = self.config.get("hyper_parameters.cat_emb_size",
                                            64)
        self.act = self.config.get("hyper_parameters.act", "sigmoid")
        self.is_sparse = self.config.get("hyper_parameters.is_sparse", False)
        self.use_DataLoader = self.config.get(
            "hyper_parameters.use_DataLoader", False)
        self.item_count = self.config.get("hyper_parameters.item_count", 63001)
        self.cat_count = self.config.get("hyper_parameters.cat_count", 801)
        self.learning_rate_base_lr = self.config.get(
            "hyper_parameters.optimizer.learning_rate_base_lr")

    def create_feeds(self, is_infer=False):
        seq_len = -1
        self.data_var = []
        hist_item_seq = paddle.static.data(
            name="hist_item_seq", shape=[None, seq_len], dtype="int64")
        self.data_var.append(hist_item_seq)

        hist_cat_seq = paddle.static.data(
            name="hist_cat_seq", shape=[None, seq_len], dtype="int64")
        self.data_var.append(hist_cat_seq)

        target_item = paddle.static.data(
            name="target_item", shape=[None], dtype="int64")
        self.data_var.append(target_item)

        target_cat = paddle.static.data(
            name="target_cat", shape=[None], dtype="int64")
        self.data_var.append(target_cat)

        label = paddle.static.data(
            name="label", shape=[-1, 1], dtype="float32")
        self.data_var.append(label)

        mask = paddle.static.data(
            name="mask", shape=[None, seq_len, 1], dtype="int64")
        self.data_var.append(mask)

        target_item_seq = paddle.static.data(
            name="target_item_seq", shape=[None, seq_len], dtype="int64")
        self.data_var.append(target_item_seq)

        target_cat_seq = paddle.static.data(
            name="target_cat_seq", shape=[None, seq_len], dtype="int64")
        self.data_var.append(target_cat_seq)

        train_inputs = [hist_item_seq] + [hist_cat_seq] + [target_item] + [
            target_cat
        ] + [label] + [mask] + [target_item_seq] + [target_cat_seq]
        return train_inputs

    def net(self, inputs, is_infer=False):
        self.hist_item_seq = inputs[0]
        self.hist_cat_seq = inputs[1]
        self.target_item = inputs[2]
        self.target_cat = inputs[3]
        self.label = inputs[4].reshape([-1, 1])
        self.mask = inputs[5]
        self.target_item_seq = inputs[6]
        self.target_cat_seq = inputs[7]
        din_model = DINLayer(self.item_emb_size, self.cat_emb_size, self.act,
                             self.is_sparse, self.use_DataLoader,
                             self.item_count, self.cat_count)

        raw_predict = din_model.forward(
            self.hist_item_seq, self.hist_cat_seq, self.target_item,
            self.target_cat, self.label, self.mask, self.target_item_seq,
            self.target_cat_seq)

        avg_loss = paddle.nn.functional.binary_cross_entropy_with_logits(
            raw_predict, self.label, reduction='mean')
        self._cost = avg_loss

        self.predict = paddle.nn.functional.sigmoid(raw_predict)
        predict_2d = paddle.concat([1 - self.predict, self.predict], 1)
        label_int = paddle.cast(self.label, 'int64')
        auc, batch_auc, _ = paddle.static.auc(input=predict_2d,
                                              label=label_int,
                                              slide_steps=0)

        self.inference_target_var = auc
        if is_infer:
            fetch_dict = {'auc': auc}
            return fetch_dict

        fetch_dict = {'cost': avg_loss, 'auc': auc}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        # optimizer = paddle.optimizer.Adam(learning_rate=self.learning_rate)
        # if strategy != None:
        #     import paddle.distributed.fleet as fleet
        #     optimizer = fleet.distributed_optimizer(optimizer, strategy)
        # optimizer.minimize(self._cost)

        boundaries = [410000]
        values = [self.learning_rate_base_lr, 0.2]
        optimizer = paddle.optimizer.SGD(
            learning_rate=paddle.optimizer.lr.PiecewiseDecay(
                boundaries=boundaries, values=values))

        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)
        return optimizer

    def infer_net(self, input):
        return self.net(input, is_infer=True)
