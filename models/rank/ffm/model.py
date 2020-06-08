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
        self.reg = envs.get_global_env("hyper_parameters.reg", 1e-4)
        self.num_field = envs.get_global_env("hyper_parameters.num_field",
                                             None)

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
            is_sparse=True,
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
        b_linear = fluid.layers.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=fluid.initializer.ConstantInitializer(
                value=0))  # 1
        # ------------------------- Field-aware second order term --------------------------

        embedding_size_for_all_field = self.num_field * self.sparse_feature_dim
        feat_embeddings_re = fluid.embedding(
            input=feat_idx,
            is_sparse=True,
            is_distributed=self.is_distributed,
            dtype='float32',
            size=[
                self.sparse_feature_number + 1, embedding_size_for_all_field
            ],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0,
                    scale=init_value_ /
                    math.sqrt(float(embedding_size_for_all_field))))
        )  # (batch_size * num_field) * 1 * embedding_size
        feat_embeddings = fluid.layers.reshape(
            feat_embeddings_re,
            shape=[-1, self.num_field, embedding_size_for_all_field
                   ])  # batch_size * num_field * embedding_size
        # batch_size * num_field * (embedding_size * num_field)
        feat_embeddings = feat_embeddings * feat_value

        field_aware_feat_embedding = fluid.layers.reshape(
            feat_embeddings,
            shape=[
                -1, self.num_field, self.num_field, self.sparse_feature_dim
            ])
        field_aware_interaction_list = []
        for i in range(self.num_field):
            for j in range(i + 1, self.num_field):
                field_aware_interaction_list.append(
                    fluid.layers.reduce_sum(
                        field_aware_feat_embedding[:, i, j, :] *
                        field_aware_feat_embedding[:, j, i, :],
                        dim=1,
                        keep_dim=True))
        y_field_aware_second_order = fluid.layers.sum(
            field_aware_interaction_list)

        # ------------------------- Predict --------------------------

        self.predict = fluid.layers.sigmoid(b_linear + y_first_order +
                                            y_field_aware_second_order)

        cost = fluid.layers.log_loss(
            input=self.predict, label=fluid.layers.cast(self.label,
                                                        "float32"))  # log_loss
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
