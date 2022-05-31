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

from net import ESCMLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.max_len = self.config.get("hyper_parameters.max_len", 3)
        self.global_w = self.config.get("hyper_parameters.global_w", 0.5)
        self.counterfactual_w = self.config.get(
            "hyper_parameters.counterfactual_w", 0.5)
        self.sparse_feature_number = self.config.get(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = self.config.get(
            "hyper_parameters.sparse_feature_dim")
        self.num_field = self.config.get("hyper_parameters.num_field")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.ctr_fc_sizes = self.config.get("hyper_parameters.ctr_fc_sizes")
        self.cvr_fc_sizes = self.config.get("hyper_parameters.cvr_fc_sizes")
        self.expert_num = self.config.get("hyper_parameters.expert_num")
        self.counterfact_mode = self.config.get("runner.counterfact_mode")
        self.expert_size = self.config.get("hyper_parameters.expert_size")
        self.tower_size = self.config.get("hyper_parameters.tower_size")
        self.feature_size = self.config.get("hyper_parameters.feature_size")

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

    def counterfact_ipw(self, loss_cvr, ctr_num, O, ctr_out_one):
        PS = paddle.multiply(
            ctr_out_one, paddle.cast(
                ctr_num, dtype="float32"))
        min_v = paddle.full_like(PS, 0.000001)
        PS = paddle.maximum(PS, min_v)
        IPS = paddle.reciprocal(PS)
        batch_shape = paddle.full_like(O, 1)
        batch_size = paddle.sum(paddle.cast(
            batch_shape, dtype="float32"),
                                axis=0)
        #TODO this shoud be a hyparameter
        IPS = paddle.clip(IPS, min=-15, max=15)  #online trick 
        IPS = paddle.multiply(IPS, batch_size)
        IPS.stop_gradient = True
        loss_cvr = paddle.multiply(loss_cvr, IPS)
        loss_cvr = paddle.multiply(loss_cvr, O)
        return paddle.mean(loss_cvr)

    def counterfact_dr(self, loss_cvr, O, ctr_out_one, imp_out):
        #dr error part
        e = paddle.subtract(loss_cvr, imp_out)

        min_v = paddle.full_like(ctr_out_one, 0.000001)
        ctr_out_one = paddle.maximum(ctr_out_one, min_v)
        IPS = paddle.divide(paddle.cast(O, dtype="float32"), ctr_out_one)

        IPS = paddle.clip(IPS, min=-15, max=15)  #online trick 
        IPS.stop_gradient = True

        loss_error_second = paddle.multiply(e, IPS)

        loss_error = imp_out + loss_error_second

        #dr imp part
        loss_imp = paddle.square(e)
        loss_imp = paddle.multiply(loss_imp, IPS)

        loss_dr = loss_error + loss_imp

        return paddle.mean(loss_dr)

    def net(self, inputs, is_infer=False):

        escm_model = ESCMLayer(
            self.sparse_feature_number, self.sparse_feature_dim,
            self.num_field, self.ctr_fc_sizes, self.cvr_fc_sizes,
            self.expert_num, self.expert_size, self.tower_size,
            self.counterfact_mode, self.feature_size)

        out_list = escm_model.forward(inputs[0:-2])

        ctr_out, ctr_out_one, cvr_out, cvr_out_one, ctcvr_prop, ctcvr_prop_one = out_list[
            0], out_list[1], out_list[2], out_list[3], out_list[4], out_list[5]
        ctr_clk = inputs[-2]
        ctcvr_buy = inputs[-1]
        ctr_num = paddle.sum(paddle.cast(ctr_clk, dtype="float32"), axis=0)
        O = paddle.cast(ctr_clk, 'float32')

        auc_ctr, batch_auc_ctr, auc_states_ctr = paddle.static.auc(
            input=ctr_out, label=ctr_clk)
        auc_ctcvr, batch_auc_ctcvr, auc_states_ctcvr = paddle.static.auc(
            input=ctcvr_prop, label=ctcvr_buy)
        auc_cvr, batch_auc_cvr, auc_states_cvr = paddle.static.auc(
            input=cvr_out, label=ctcvr_buy)

        if is_infer:
            fetch_dict = {
                'auc_ctr': auc_ctr,
                'auc_cvr': auc_cvr,
                'auc_ctcvr': auc_ctcvr
            }
            return fetch_dict

        loss_ctr = paddle.nn.functional.log_loss(
            input=ctr_out_one, label=paddle.cast(
                ctr_clk, dtype="float32"))
        loss_cvr = paddle.nn.functional.log_loss(
            input=cvr_out_one, label=paddle.cast(
                ctcvr_buy, dtype="float32"))
        if self.counterfact_mode == "DR":
            loss_cvr = self.counterfact_dr(loss_cvr, O, ctr_out_one,
                                           out_list[6])
        else:
            loss_cvr = self.counterfact_ipw(loss_cvr, ctr_num, O, ctr_out_one)
        loss_ctcvr = paddle.nn.functional.log_loss(
            input=ctcvr_prop_one,
            label=paddle.cast(
                ctcvr_buy, dtype="float32"))
        cost = loss_ctr + loss_cvr * self.counterfactual_w + loss_ctcvr * self.global_w
        avg_cost = paddle.mean(x=cost)

        self._cost = avg_cost
        fetch_dict = {
            'cost': avg_cost,
            'auc_ctr': auc_ctr,
            'auc_cvr': auc_cvr,
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
