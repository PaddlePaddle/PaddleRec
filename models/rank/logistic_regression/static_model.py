#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from net import LRLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.sparse_feature_number = self.config.get(
            "hyper_parameters.sparse_feature_number", None)
        self.num_field = self.config.get("hyper_parameters.num_field", None)
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.reg = self.config.get("hyper_parameters.reg", 1e-4)

    def create_feeds(self, is_infer=False):
        dense_input = paddle.static.data(
            name="dense_input", shape=[None, self.num_field], dtype="float32")

        sparse_input_ids = paddle.static.data(
            name="sparse_input", shape=[None, self.num_field], dtype="int64")

        label = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64")

        feeds_list = [label] + [sparse_input_ids] + [dense_input]
        return feeds_list

    def net(self, inputs, is_infer=False):
        init_value_ = 0.1
        # ------------------------- network input --------------------------

        self.label = inputs[0]
        feat_idx = inputs[1]
        feat_value = inputs[2]

        #feat_value = paddle.reshape(
        #    raw_feat_value, [-1, self.num_field])  # None * num_field * 1

        LR_model = LRLayer(self.sparse_feature_number, init_value_, self.reg,
                           self.num_field)

        self.predict = LR_model.forward(feat_idx, feat_value)

        predict_2d = paddle.concat(x=[1 - self.predict, self.predict], axis=1)
        label_int = paddle.cast(self.label, 'int64')
        auc, batch_auc_var, _ = paddle.static.auc(input=predict_2d,
                                                  label=label_int,
                                                  slide_steps=0)
        self.inference_target_var = auc
        if is_infer:
            fetch_dict = {'auc': auc}
            return fetch_dict
        cost = paddle.nn.functional.log_loss(
            input=self.predict, label=paddle.cast(self.label, "float32"))
        avg_cost = paddle.sum(x=cost)
        self._cost = avg_cost
        fetch_dict = {'cost': avg_cost, 'auc': auc}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adam(
            learning_rate=self.learning_rate, lazy_mode=True)
        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)
