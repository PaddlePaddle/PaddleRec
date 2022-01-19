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
from net import DSSMLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.trigram_d = self.config.get("hyper_parameters.trigram_d")
        self.neg_num = self.config.get("hyper_parameters.neg_num")
        self.hidden_layers = self.config.get("hyper_parameters.fc_sizes")
        self.hidden_acts = self.config.get("hyper_parameters.fc_acts")
        self.learning_rate = self.config.get("hyper_parameters.learning_rate")
        self.slice_end = self.config.get("hyper_parameters.slice_end")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")

    def create_feeds(self, is_infer=False):
        query = paddle.static.data(
            name="query", shape=[-1, self.trigram_d], dtype='float32')
        self.prune_feed_vars = [query]

        doc_pos = paddle.static.data(
            name="doc_pos", shape=[-1, self.trigram_d], dtype='float32')

        if is_infer:
            return [query, doc_pos]

        doc_negs = [
            paddle.static.data(
                name="doc_neg_" + str(i),
                shape=[-1, self.trigram_d],
                dtype="float32") for i in range(self.neg_num)
        ]
        feeds_list = [query, doc_pos] + doc_negs
        return feeds_list

    def net(self, input, is_infer=False):
        dssm_model = DSSMLayer(self.trigram_d, self.neg_num, self.slice_end,
                               self.hidden_layers, self.hidden_acts)
        R_Q_D_p, hit_prob = dssm_model.forward(input, is_infer)

        self.inference_target_var = R_Q_D_p
        self.prune_target_var = dssm_model.query_fc
        self.train_dump_fields = [dssm_model.query_fc, R_Q_D_p]
        self.train_dump_params = dssm_model.params
        self.infer_dump_fields = [dssm_model.doc_pos_fc]
        if is_infer:
            fetch_dict = {'query_doc_sim': R_Q_D_p}
            return fetch_dict
        loss = -paddle.sum(paddle.log(hit_prob), axis=-1)
        avg_cost = paddle.mean(x=loss)
        # print(avg_cost)
        self._cost = avg_cost
        fetch_dict = {'Loss': avg_cost}
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
