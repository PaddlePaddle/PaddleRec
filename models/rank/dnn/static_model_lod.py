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

from net import DNNLayer, StaticDNNLayer


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

        self.sparse_feature_number = self.config.get(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = self.config.get(
            "hyper_parameters.sparse_feature_dim")
        self.sparse_inputs_slots = self.config.get(
            "hyper_parameters.sparse_inputs_slots")
        self.dense_input_dim = self.config.get(
            "hyper_parameters.dense_input_dim")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.fc_sizes = self.config.get("hyper_parameters.fc_sizes")

    def create_feeds(self, is_infer=False):
        dense_input = paddle.static.data(
            name="dense_input",
            shape=[None, self.dense_input_dim],
            dtype="float32")

        sparse_input_ids = [
            paddle.static.data(
                name="C" + str(i), shape=[None, 1], lod_level=1, dtype="int64")
            for i in range(1, self.sparse_inputs_slots)
        ]

        label = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64")

        feeds_list = [label] + sparse_input_ids + [dense_input]
        return feeds_list

    def net(self, input, is_infer=False):
        self.label_input = input[0]
        self.sparse_inputs = input[1:self.sparse_inputs_slots]
        self.dense_input = input[-1]
        sparse_number = self.sparse_inputs_slots - 1

        def embedding_layer(input):
            if self.distributed_embedding:
                emb = paddle.static.nn.sparse_embedding(
                    input=input,
                    size=[
                        self.sparse_feature_number, self.sparse_feature_dim
                    ],
                    param_attr=paddle.ParamAttr(
                        name="SparseFeatFactors",
                        initializer=paddle.nn.initializer.Uniform()))
            else:
                paddle.static.Print(input)

                emb = paddle.static.nn.embedding(
                    input=input,
                    is_sparse=True,
                    is_distributed=self.is_distributed,
                    size=[
                        self.sparse_feature_number, self.sparse_feature_dim
                    ],
                    param_attr=paddle.ParamAttr(
                        name="SparseFeatFactors",
                        initializer=paddle.initializer.Uniform()))
            emb_sum = paddle.static.nn.sequence_pool(
                input=emb, pool_type='sum')
            return emb_sum

        sparse_embs = list(map(embedding_layer, self.sparse_inputs))

        dnn_model = StaticDNNLayer(
            self.sparse_feature_number, self.sparse_feature_dim,
            self.dense_input_dim, sparse_number, self.fc_sizes)

        raw_predict_2d = dnn_model.forward(sparse_embs, self.dense_input)

        predict_2d = paddle.nn.functional.softmax(raw_predict_2d)

        self.predict = predict_2d

        auc, batch_auc, _ = paddle.static.auc(input=self.predict,
                                              label=self.label_input,
                                              num_thresholds=2**12,
                                              slide_steps=20)
        self.inference_target_var = auc
        if is_infer:
            fetch_dict = {'auc': auc}
            return fetch_dict

        cost = paddle.nn.functional.cross_entropy(
            input=raw_predict_2d, label=self.label_input)
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
