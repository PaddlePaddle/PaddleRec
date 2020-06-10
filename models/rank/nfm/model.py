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
from collections import OrderedDict

import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.is_distributed = True if envs.get_fleet_mode().upper(
        ) == "PSLIB" else False
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number", None)
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim", None)
        self.is_sparse = envs.get_global_env("hyper_parameters.is_sparse",
                                             False)
        self.use_batchnorm = envs.get_global_env(
            "hyper_parameters.use_batchnorm", False)
        self.use_dropout = envs.get_global_env("hyper_parameters.use_dropout",
                                               False)
        self.dropout_prob = envs.get_global_env(
            "hyper_parameters.dropout_prob", None)
        self.layer_sizes = envs.get_global_env("hyper_parameters.fc_sizes",
                                               None)
        self.loss_type = envs.get_global_env("hyper_parameters.loss_type",
                                             'logloss')
        self.reg = envs.get_global_env("hyper_parameters.reg", 1e-4)
        self.num_field = envs.get_global_env("hyper_parameters.num_field",
                                             None)
        self.act = envs.get_global_env("hyper_parameters.act", None)

    def net(self, inputs, is_infer=False):
        raw_feat_idx = self._sparse_data_var[1]  # (batch_size * num_field) * 1
        raw_feat_value = self._dense_data_var[0]  # batch_size * num_field
        self.label = self._sparse_data_var[0]  # batch_size * 1

        init_value_ = 0.1

        feat_idx = raw_feat_idx
        feat_value = fluid.layers.reshape(
            raw_feat_value,
            [-1, self.num_field, 1])  # batch_size * num_field * 1

        # ------------------------- first order term --------------------------

        first_weights_re = fluid.embedding(
            input=feat_idx,
            is_sparse=self.is_sparse,
            is_distributed=self.is_distributed,
            dtype='float32',
            size=[self.sparse_feature_number + 1, 1],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_),
                regularizer=fluid.regularizer.L1DecayRegularizer(self.reg))
        )  # (batch_size * num_field) * 1 * 1(embedding_size)
        first_weights = fluid.layers.reshape(
            first_weights_re,
            shape=[-1, self.num_field, 1])  # batch_size * num_field * 1
        y_first_order = fluid.layers.reduce_sum((first_weights * feat_value),
                                                1)  # batch_size * 1
        # ------------------------- second order term --------------------------

        feat_embeddings_re = fluid.embedding(
            input=feat_idx,
            is_sparse=self.is_sparse,
            is_distributed=self.is_distributed,
            dtype='float32',
            size=[self.sparse_feature_number + 1, self.sparse_feature_dim],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0,
                    scale=init_value_ /
                    math.sqrt(float(self.sparse_feature_dim))))
        )  # (batch_size * num_field) * 1 * embedding_size
        feat_embeddings = fluid.layers.reshape(
            feat_embeddings_re,
            shape=[-1, self.num_field, self.sparse_feature_dim
                   ])  # batch_size * num_field * embedding_size
        # batch_size * num_field * embedding_size
        feat_embeddings = feat_embeddings * feat_value

        # sum_square part
        summed_features_emb = fluid.layers.reduce_sum(
            feat_embeddings, 1)  # batch_size * embedding_size
        summed_features_emb_square = fluid.layers.square(
            summed_features_emb)  # batch_size * embedding_size

        # square_sum part
        squared_features_emb = fluid.layers.square(
            feat_embeddings)  # batch_size * num_field * embedding_size
        squared_sum_features_emb = fluid.layers.reduce_sum(
            squared_features_emb, 1)  # batch_size * embedding_size

        y_FM = 0.5 * (summed_features_emb_square - squared_sum_features_emb
                      )  # batch_size * embedding_size

        if self.use_batchnorm:
            y_FM = fluid.layers.batch_norm(input=y_FM, is_test=is_infer)
        if self.use_dropout:
            y_FM = fluid.layers.dropout(
                x=y_FM, dropout_prob=self.dropout_prob, is_test=is_infer)

        # ------------------------- DNN --------------------------

        y_dnn = y_FM
        for s in self.layer_sizes:
            if self.use_batchnorm:
                y_dnn = fluid.layers.fc(
                    input=y_dnn,
                    size=s,
                    act=self.act,
                    param_attr=fluid.ParamAttr(initializer=fluid.initializer.
                                               TruncatedNormalInitializer(
                                                   loc=0.0,
                                                   scale=init_value_ /
                                                   math.sqrt(float(10)))),
                    bias_attr=fluid.ParamAttr(initializer=fluid.initializer.
                                              TruncatedNormalInitializer(
                                                  loc=0.0, scale=init_value_)))
                y_dnn = fluid.layers.batch_norm(
                    input=y_dnn, act=self.act, is_test=is_infer)
            else:
                y_dnn = fluid.layers.fc(
                    input=y_dnn,
                    size=s,
                    act=self.act,
                    param_attr=fluid.ParamAttr(initializer=fluid.initializer.
                                               TruncatedNormalInitializer(
                                                   loc=0.0,
                                                   scale=init_value_ /
                                                   math.sqrt(float(10)))),
                    bias_attr=fluid.ParamAttr(initializer=fluid.initializer.
                                              TruncatedNormalInitializer(
                                                  loc=0.0, scale=init_value_)))
            if self.use_dropout:
                y_dnn = fluid.layers.dropout(
                    x=y_dnn, dropout_prob=self.dropout_prob, is_test=is_infer)
        y_dnn = fluid.layers.fc(
            input=y_dnn,
            size=1,
            act=None,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_)))

        # ------------------------- Predict --------------------------

        self.predict = fluid.layers.sigmoid(y_first_order + y_dnn)
        if self.loss_type == "squqre_loss":
            cost = fluid.layers.mse_loss(
                input=self.predict,
                label=fluid.layers.cast(self.label, "float32"))
        else:
            cost = fluid.layers.log_loss(
                input=self.predict,
                label=fluid.layers.cast(self.label,
                                        "float32"))  # default log_loss
        avg_cost = fluid.layers.reduce_sum(cost)

        self._cost = avg_cost

        predict_2d = fluid.layers.concat([1 - self.predict, self.predict], 1)
        label_int = fluid.layers.cast(self.label, 'int64')
        auc_var, batch_auc_var, _ = fluid.layers.auc(input=predict_2d,
                                                     label=label_int,
                                                     slide_steps=0)
        self._metrics["AUC"] = auc_var
        self._metrics["BATCH_AUC"] = batch_auc_var
        if is_infer:
            self._infer_results["AUC"] = auc_var
