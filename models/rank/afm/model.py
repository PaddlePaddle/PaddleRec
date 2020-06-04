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
from paddlerec.core.model import Model as ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.is_distributed = True if envs.get_trainer(
        ) == "CtrTrainer" else False
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number", None)
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim", None)
        self.is_sparse = envs.get_global_env("hyper_parameters.is_sparse",
                                             False)
        self.reg = envs.get_global_env("hyper_parameters.reg", 1e-4)
        self.num_field = envs.get_global_env("hyper_parameters.num_field",
                                             None)
        self.hidden1_attention_size = envs.get_global_env(
            "hyper_parameters.hidden1_attention_size", 16)
        self.attention_act = envs.get_global_env("hyper_parameters.act",
                                                 "relu")

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

        # ------------------------- Pair-wise Interaction Layer --------------------------

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
        feat_embeddings = feat_embeddings * feat_value  # batch_size * num_field * embedding_size

        element_wise_product_list = []
        for i in range(self.num_field):
            for j in range(i + 1, self.num_field):
                element_wise_product_list.append(
                    feat_embeddings[:, i, :] *
                    feat_embeddings[:,
                                    j, :])  # list(batch_size * embedding_size)
        stack_element_wise_product = fluid.layers.stack(
            element_wise_product_list,
            axis=0)  # (num_field*(num_field-1)/2) * batch_size * embedding_size
        stack_element_wise_product = fluid.layers.transpose(
            stack_element_wise_product, perm=[1, 0, 2]
        )  # batch_size * (num_field*(num_field-1)/2) * embedding_size

        # ------------------------- Attention-based Pooling --------------------------

        attetion_mul = fluid.layers.fc(
            input=fluid.layers.reshape(
                stack_element_wise_product,
                shape=[-1, self.sparse_feature_dim]),
            size=self.hidden1_attention_size,
            act=self.attention_act,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_))
        )  # (batch_size * (num_field*(num_field-1)/2)) * hidden1_attention_size
        attention_h = fluid.layers.create_parameter(
            shape=[self.hidden1_attention_size, 1], dtype="float32")

        attention_out = fluid.layers.matmul(
            attetion_mul,
            attention_h)  # (batch_size * (num_field*(num_field-1)/2)) * 1
        attention_out = fluid.layers.softmax(
            attention_out)  # (batch_size * (num_field*(num_field-1)/2)) * 1
        num_interactions = self.num_field * (self.num_field - 1) / 2
        attention_out = fluid.layers.reshape(
            attention_out,
            shape=[-1, num_interactions,
                   1])  # batch_size * (num_field*(num_field-1)/2) * 1
        attention_pooling = fluid.layers.matmul(
            attention_out, stack_element_wise_product,
            transpose_x=True)  # batch_size * 1 * embedding_size
        attention_pooling = fluid.layers.reshape(
            attention_pooling,
            shape=[-1, self.sparse_feature_dim])  # batch_size * embedding_size
        y_AFM = fluid.layers.fc(
            input=attention_pooling,
            size=1,
            act=None,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_)))  # batch_size * 1

        # ------------------------- Predict --------------------------

        self.predict = fluid.layers.sigmoid(y_first_order + y_AFM)

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
