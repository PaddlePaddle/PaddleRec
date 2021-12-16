# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from net import NAMLLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.infer_target_var = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.article_content_size = self.config.get(
            "hyper_parameters.article_content_size")
        self.article_title_size = self.config.get(
            "hyper_parameters.article_title_size")
        self.browse_size = self.config.get("hyper_parameters.browse_size")
        self.neg_condidate_sample_size = self.config.get(
            "hyper_parameters.neg_condidate_sample_size")
        self.word_dimension = self.config.get(
            "hyper_parameters.word_dimension")
        self.category_size = self.config.get("hyper_parameters.category_size")
        self.sub_category_size = self.config.get(
            "hyper_parameters.sub_category_size")
        self.cate_dimension = self.config.get(
            "hyper_parameters.category_dimension")
        self.word_dict_size = self.config.get(
            "hyper_parameters.word_dict_size")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.sample_size = self.neg_condidate_sample_size + 1

    def create_feeds(self, is_infer=False):
        inputs = [
            paddle.static.data(
                name="sampe_cate",
                shape=[None, self.sample_size],
                dtype='int64'), paddle.static.data(
                    name="browse_cate",
                    shape=[None, self.browse_size],
                    dtype='int64'), paddle.static.data(
                        name="sampe_sub_cate",
                        shape=[None, self.sample_size],
                        dtype='int64'), paddle.static.data(
                            name="browse_sub_cate",
                            shape=[None, self.browse_size],
                            dtype='int64'),
            paddle.static.data(
                name="sampe_title",
                shape=[None, self.sample_size, self.article_title_size],
                dtype='int64'), paddle.static.data(
                    name="browse_title",
                    shape=[None, self.browse_size, self.article_title_size],
                    dtype='int64'),
            paddle.static.data(
                name="sample_article",
                shape=[None, self.sample_size, self.article_content_size],
                dtype='int64'), paddle.static.data(
                    name="browse_article",
                    shape=[None, self.browse_size, self.article_content_size],
                    dtype='int64')
        ]
        label = paddle.static.data(
            name="label", shape=[None, self.sample_size], dtype="int64")
        return [label] + inputs

    def net(self, input, is_infer=False):
        self.labels = input[0]
        self.sparse_inputs = input[1:]
        #self.dense_input = input[-1]
        #sparse_number = self.sparse_inputs_slots - 1
        model = NAMLLayer(self.article_content_size, self.article_title_size,
                          self.browse_size, self.neg_condidate_sample_size,
                          self.word_dimension, self.category_size,
                          self.sub_category_size, self.cate_dimension,
                          self.word_dict_size)

        raw = model.forward(self.sparse_inputs)

        soft_predict = paddle.nn.functional.sigmoid(
            paddle.reshape(raw, [-1, 1]))
        predict_2d = paddle.concat(x=[1 - soft_predict, soft_predict], axis=-1)
        labels = paddle.reshape(self.labels, [-1, 1])
        #metrics_list[0].update(preds=predict_2d.numpy(), labels=labels.numpy())
        #self.predict = predict_2d

        auc, batch_auc, _ = paddle.static.auc(input=predict_2d,
                                              label=labels,
                                              num_thresholds=2**12,
                                              slide_steps=20)
        self.inference_target_var = auc
        if is_infer:
            fetch_dict = {'auc': auc}
            return fetch_dict

        cost = paddle.nn.functional.cross_entropy(
            input=raw,
            label=paddle.cast(self.labels, "float32"),
            soft_label=True)
        avg_cost = paddle.mean(x=cost)
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
