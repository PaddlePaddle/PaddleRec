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

from net import DMRLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.user_size = self.config.get("hyper_parameters.user_size")
        self.cms_segid_size = self.config.get(
            "hyper_parameters.cms_segid_size")
        self.cms_group_id_size = self.config.get(
            "hyper_parameters.cms_group_id_size")
        self.final_gender_code_size = self.config.get(
            "hyper_parameters.final_gender_code_size")
        self.age_level_size = self.config.get(
            "hyper_parameters.age_level_size")
        self.pvalue_level_size = self.config.get(
            "hyper_parameters.pvalue_level_size")
        self.shopping_level_size = self.config.get(
            "hyper_parameters.shopping_level_size")
        self.occupation_size = self.config.get(
            "hyper_parameters.occupation_size")
        self.new_user_class_level_size = self.config.get(
            "hyper_parameters.new_user_class_level_size")
        self.adgroup_id_size = self.config.get(
            "hyper_parameters.adgroup_id_size")
        self.cate_size = self.config.get("hyper_parameters.cate_size")
        self.campaign_id_size = self.config.get(
            "hyper_parameters.campaign_id_size")
        self.customer_size = self.config.get("hyper_parameters.customer_size")
        self.brand_size = self.config.get("hyper_parameters.brand_size")
        self.btag_size = self.config.get("hyper_parameters.btag_size")
        self.pid_size = self.config.get("hyper_parameters.pid_size")
        self.main_embedding_size = self.config.get(
            "hyper_parameters.main_embedding_size")
        self.other_embedding_size = self.config.get(
            "hyper_parameters.other_embedding_size")

        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate", 0.008)

    def create_feeds(self, is_infer=False):
        input = paddle.static.data(
            name="sparse_tensor", shape=[None, 267], dtype="float32")
        feed_list = [input]
        return feed_list

    def net(self, input, is_infer=False):
        sparse = paddle.cast(input[0], "int64")
        price = paddle.slice(input[0], [1], [264], [265])
        label = paddle.slice(input[0], [1], [266], [267])
        label = paddle.cast(label, "int64")
        inputs = [sparse, price]

        DMR_model = DMRLayer(
            self.user_size, self.cms_segid_size, self.cms_group_id_size,
            self.final_gender_code_size, self.age_level_size,
            self.pvalue_level_size, self.shopping_level_size,
            self.occupation_size, self.new_user_class_level_size,
            self.adgroup_id_size, self.cate_size, self.campaign_id_size,
            self.customer_size, self.brand_size, self.btag_size, self.pid_size,
            self.main_embedding_size, self.other_embedding_size)

        pred, loss = DMR_model.forward(inputs, is_infer)

        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        auc, batch_auc, _ = paddle.static.auc(input=predict_2d,
                                              label=label,
                                              num_thresholds=2**12,
                                              slide_steps=20)
        auc = paddle.cast(auc, "float32")

        if is_infer:
            fetch_dict = {"auc": auc}
            return fetch_dict

        self._cost = loss
        fetch_dict = {'auc': auc, 'cost': loss}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adam(
            learning_rate=self.learning_rate, lazy_mode=False)
        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)
