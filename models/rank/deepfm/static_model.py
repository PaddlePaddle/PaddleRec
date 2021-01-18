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
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.nn.functional as F
import math


class Model(object):
    def __init__(self, config):
        self.cost = None
        self.metrics = {}
        self.config = config
        self.init_hyper_parameters()

    def init_hyper_parameters(self):
        self.sparse_feature_number = self.config.get(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = self.config.get(
            "hyper_parameters.sparse_feature_dim")
        self.sparse_inputs_slot = self.config.get(
            "hyper_parameters.sparse_inputs_slots")
        self.dense_input_dim = self.config.get(
            "hyper_parameters.dense_input_dim")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.fc_sizes = self.config.get("hyper_parameters.fc_sizes")
        self.adam_lazy_mode = self.config.get(
            "hyper_parameters.optimizer.adam_lazy_mode")

    def input_data(self):
        dense_input = fluid.layers.data(
            name="dense_input", shape=[self.dense_input_dim], dtype="float32")

        sparse_input_ids = [
            fluid.layers.data(
                name="C" + str(i), shape=[1], lod_level=1, dtype="int64")
            for i in range(1, self.sparse_inputs_slot)
        ]

        label = fluid.layers.data(name="label", shape=[1], dtype="int64")

        inputs = [dense_input] + sparse_input_ids + [label]
        return inputs

    def net(self, inputs, is_infer=False):
        self.sparse_inputs = inputs[1:-1]
        self.dense_input = inputs[0]
        self.label_input = inputs[-1]

        deepfm_model = DeepFMLayer(
            self.sparse_feature_number, self.sparse_feature_dim,
            self.dense_input_dim, self.sparse_inputs_slot - 1, self.fc_sizes)

        pred = deepfm_model(self.sparse_inputs, self.dense_input)

        #pred = F.sigmoid(prediction)

        predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)

        auc, batch_auc, _ = paddle.fluid.layers.auc(input=predict_2d,
                                                    label=self.label_input,
                                                    num_thresholds=2**12,
                                                    slide_steps=20)

        if is_infer:
            self._infer_results["AUC"] = auc
            return

        cost = paddle.nn.functional.log_loss(
            input=pred, label=paddle.cast(
                self.label_input, dtype="float32"))
        avg_cost = paddle.mean(x=cost)
        self.cost = avg_cost
        self.infer_target_var = auc
        return {'cost': avg_cost, 'auc': auc}

    def minimize(self, strategy=None):
        optimizer = paddle.optimizer.Adam(
            self.learning_rate, lazy_mode=self.adam_lazy_mode)
        if strategy != None:
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self.cost)

    def infer_net(self):
        pass


class DeepFMLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field, layer_sizes):
        super(DeepFMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.layer_sizes = layer_sizes

        self.fm = FM(sparse_feature_number, sparse_feature_dim,
                     dense_feature_dim, sparse_num_field)
        self.dnn = DNN(sparse_feature_number, sparse_feature_dim,
                       dense_feature_dim, dense_feature_dim + sparse_num_field,
                       layer_sizes)
        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, sparse_inputs, dense_inputs):

        y_first_order, y_second_order, feat_embeddings = self.fm(sparse_inputs,
                                                                 dense_inputs)
        y_dnn = self.dnn(feat_embeddings)

        predict = F.sigmoid(y_first_order + y_second_order + y_dnn)

        return predict


class FM(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field):
        super(FM, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.dense_emb_dim = self.sparse_feature_dim
        self.sparse_num_field = sparse_num_field
        self.init_value_ = 0.1

        self.embedding_one = paddle.nn.Embedding(
            sparse_feature_number,
            1,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        # dense coding
        self.dense_w_one = paddle.create_parameter(
            shape=[self.dense_feature_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0))

        self.dense_w = paddle.create_parameter(
            shape=[1, self.dense_feature_dim, self.dense_emb_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0))

    def forward(self, sparse_inputs, dense_inputs):
        # -------------------- first order term  --------------------
        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)
        sparse_emb_one = self.embedding_one(sparse_inputs_concat)

        dense_emb_one = paddle.multiply(dense_inputs, self.dense_w_one)
        dense_emb_one = paddle.unsqueeze(dense_emb_one, axis=2)

        y_first_order = paddle.sum(sparse_emb_one, 1) + paddle.sum(
            dense_emb_one, 1)

        # -------------------- second order term  --------------------
        sparse_embeddings = self.embedding(sparse_inputs_concat)
        dense_inputs_re = paddle.unsqueeze(dense_inputs, axis=2)
        dense_embeddings = paddle.multiply(dense_inputs_re, self.dense_w)
        feat_embeddings = paddle.concat([sparse_embeddings, dense_embeddings],
                                        1)

        # sum_square part
        summed_features_emb = paddle.sum(feat_embeddings,
                                         1)  # None * embedding_size
        summed_features_emb_square = paddle.square(
            summed_features_emb)  # None * embedding_size

        # square_sum part
        squared_features_emb = paddle.square(
            feat_embeddings)  # None * num_field * embedding_size
        squared_sum_features_emb = paddle.sum(squared_features_emb,
                                              1)  # None * embedding_size

        y_second_order = 0.5 * paddle.sum(
            summed_features_emb_square - squared_sum_features_emb,
            1,
            keepdim=True)  # None * 1

        return y_first_order, y_second_order, feat_embeddings


class DNN(paddle.nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes):
        super(DNN, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = num_field
        self.layer_sizes = layer_sizes

        # sizes = [sparse_feature_dim * num_field + dense_feature_dim
        sizes = [sparse_feature_dim * num_field] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)

    def forward(self, feat_embeddings):
        y_dnn = paddle.reshape(feat_embeddings,
                               [-1, self.num_field * self.sparse_feature_dim])
        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
        return y_dnn
