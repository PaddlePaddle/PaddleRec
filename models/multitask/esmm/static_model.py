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

import numpy as np
import paddle

from net import ESMMLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.max_len = self.config.get("hyper_parameters.max_len", 3)
        self.sparse_feature_number = self.config.get(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = self.config.get(
            "hyper_parameters.sparse_feature_dim")
        self.num_field = self.config.get("hyper_parameters.num_field")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.ctr_fc_sizes = self.config.get("hyper_parameters.ctr_fc_sizes")
        self.cvr_fc_sizes = self.config.get("hyper_parameters.cvr_fc_sizes")

    def create_feeds(self, is_infer=False):
        sparse_input_ids = [
            paddle.static.data(
                name="field_" + str(i),
                shape=[None, self.max_len],
                dtype="int64") for i in range(0, 23)
        ]
        label_ctr = paddle.static.data(
            name="ctr", shape=[None, 1], dtype="int64")
        label_cvr = paddle.static.data(
            name="cvr", shape=[None, 1], dtype="int64")
        inputs = sparse_input_ids + [label_ctr] + [label_cvr]
        if is_infer:
            return inputs
        else:
            return inputs

    def net(self, inputs, is_infer=False):

        esmm_model = ESMMLayer(self.sparse_feature_number,
                               self.sparse_feature_dim, self.num_field,
                               self.ctr_fc_sizes, self.cvr_fc_sizes)

        ctr_out, ctr_out_one, cvr_out, cvr_out_one, ctcvr_prop, ctcvr_prop_one = esmm_model.forward(
            inputs[0:-2])

        ctr_clk = inputs[-2]
        ctcvr_buy = inputs[-1]

        auc_ctr, batch_auc_ctr, auc_states_ctr = paddle.static.auc(
            input=ctr_out, label=ctr_clk)
        auc_ctcvr, batch_auc_ctcvr, auc_states_ctcvr = paddle.static.auc(
            input=ctcvr_prop, label=ctcvr_buy)

        if is_infer:
            fetch_dict = {'auc_ctr': auc_ctr, 'auc_ctcvr': auc_ctcvr}
            return fetch_dict

        loss_ctr = paddle.nn.functional.log_loss(
            input=ctr_out_one, label=paddle.cast(
                ctr_clk, dtype="float32"))
        loss_ctcvr = paddle.nn.functional.log_loss(
            input=ctcvr_prop_one,
            label=paddle.cast(
                ctcvr_buy, dtype="float32"))
        cost = loss_ctr + loss_ctcvr
        avg_cost = paddle.mean(x=cost)

        self._cost = avg_cost
        fetch_dict = {
            'cost': avg_cost,
            'auc_ctr': auc_ctr,
            'auc_ctcvr': auc_ctcvr
        }
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
