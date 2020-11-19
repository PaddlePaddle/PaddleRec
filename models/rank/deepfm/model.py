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

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number", None)
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim", None)
        self.num_field = envs.get_global_env("hyper_parameters.num_field",
                                             None)
        self.reg = envs.get_global_env("hyper_parameters.reg", 1e-4)
        self.layer_sizes = envs.get_global_env("hyper_parameters.fc_sizes",
                                               None)
        self.act = envs.get_global_env("hyper_parameters.act", None)

    def net(self, inputs, is_infer=False):
        init_value_ = 0.1
        is_distributed = True if envs.get_trainer() == "CtrTrainer" else False

        # ------------------------- network input --------------------------

        raw_feat_idx = self._sparse_data_var[1]
        raw_feat_value = self._dense_data_var[0]
        self.label = self._sparse_data_var[0]

        feat_idx = raw_feat_idx
        feat_value = paddle.fluid.layers.nn.reshape(
            raw_feat_value, [-1, self.num_field, 1])  # None * num_field * 1

        first_weights_re = paddle.static.nn.embedding(
            input=feat_idx,
            is_sparse=True,
            is_distributed=is_distributed,
            dtype='float32',
            size=[self.sparse_feature_number + 1, 1],
            padding_idx=0,
            param_attr=paddle.ParamAttr(
                initializer=paddle.fluid.initializer.
                TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_),
                regularizer=paddle.regularizer.L1Decay(coeff=self.reg)))
        first_weights = paddle.fluid.layers.nn.reshape(
            first_weights_re,
            shape=[-1, self.num_field, 1])  # None * num_field * 1
        y_first_order = paddle.sum(x=(first_weights * feat_value), axis=1)

        # ------------------------- second order term --------------------------

        feat_embeddings_re = paddle.static.nn.embedding(
            input=feat_idx,
            is_sparse=True,
            is_distributed=is_distributed,
            dtype='float32',
            size=[self.sparse_feature_number + 1, self.sparse_feature_dim],
            padding_idx=0,
            param_attr=paddle.ParamAttr(
                initializer=paddle.fluid.initializer.
                TruncatedNormalInitializer(
                    loc=0.0,
                    scale=init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))
        feat_embeddings = paddle.fluid.layers.nn.reshape(
            feat_embeddings_re,
            shape=[-1, self.num_field, self.sparse_feature_dim
                   ])  # None * num_field * embedding_size
        # None * num_field * embedding_size
        feat_embeddings = feat_embeddings * feat_value

        # sum_square part
        summed_features_emb = paddle.sum(x=feat_embeddings,
                                         axis=1)  # None * embedding_size
        summed_features_emb_square = paddle.square(
            summed_features_emb)  # None * embedding_size

        # square_sum part
        squared_features_emb = paddle.square(
            feat_embeddings)  # None * num_field * embedding_size
        squared_sum_features_emb = paddle.sum(x=squared_features_emb,
                                              axis=1)  # None * embedding_size

        y_second_order = 0.5 * paddle.sum(
            x=summed_features_emb_square - squared_sum_features_emb,
            axis=1,
            keepdim=True)  # None * 1

        # ------------------------- DNN --------------------------

        y_dnn = paddle.fluid.layers.nn.reshape(
            feat_embeddings, [-1, self.num_field * self.sparse_feature_dim])
        for s in self.layer_sizes:
            y_dnn = paddle.static.nn.fc(
                x=y_dnn,
                size=s,
                activation=self.act,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.fluid.initializer.
                    TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_ / math.sqrt(float(10)))),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.fluid.initializer.
                    TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_)))
        y_dnn = paddle.static.nn.fc(
            x=y_dnn,
            size=1,
            activation=None,
            weight_attr=paddle.ParamAttr(initializer=paddle.fluid.initializer.
                                         TruncatedNormalInitializer(
                                             loc=0.0, scale=init_value_)),
            bias_attr=paddle.ParamAttr(initializer=paddle.fluid.initializer.
                                       TruncatedNormalInitializer(
                                           loc=0.0, scale=init_value_)))

        # ------------------------- DeepFM --------------------------

        self.predict = paddle.nn.functional.sigmoid(y_first_order +
                                                    y_second_order + y_dnn)
        cost = paddle.nn.functional.log_loss(
            input=self.predict, label=paddle.cast(self.label, "float32"))
        avg_cost = paddle.sum(x=cost)

        self._cost = avg_cost

        predict_2d = paddle.concat(x=[1 - self.predict, self.predict], axis=1)
        label_int = paddle.cast(self.label, 'int64')
        auc_var, batch_auc_var, _ = paddle.fluid.layers.auc(input=predict_2d,
                                                            label=label_int,
                                                            slide_steps=0)
        self._metrics["AUC"] = auc_var
        self._metrics["BATCH_AUC"] = batch_auc_var
        if is_infer:
            self._infer_results["AUC"] = auc_var
