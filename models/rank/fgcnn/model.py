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
        self.filters = envs.get_global_env("hyper_parameters.filters",
                                           [38, 40, 42, 44])
        self.filter_size = envs.get_global_env("hyper_parameters.filter_size",
                                               [1, 9])
        self.pooling_size = envs.get_global_env(
            "hyper_parameters.pooling_size", [2, 2, 2, 2])
        self.new_filters = envs.get_global_env("hyper_parameters.new_filters",
                                               [3, 3, 3, 3])
        self.hidden_layers = envs.get_global_env("hyper_parameters.fc_sizes")
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

        # ------------------------- Embedding layers --------------------------

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
        featuer_generation_input = fluid.layers.reshape(
            feat_embeddings,
            shape=[0, 1, self.num_field, self.sparse_feature_dim])
        new_feature_list = []
        new_feature_field_num = 0

        # ------------------------- Feature Generation --------------------------

        for i in range(len(self.filters)):
            conv_out = fluid.layers.conv2d(
                featuer_generation_input,
                num_filters=self.filters[i],
                filter_size=self.filter_size,
                padding="SAME",
                act="tanh")
            pool_out = fluid.layers.pool2d(
                conv_out,
                pool_size=[self.pooling_size[i], 1],
                pool_type="max",
                pool_stride=[self.pooling_size[i], 1])
            pool_out_shape = pool_out.shape[2]
            new_feature_field_num += self.new_filters[i] * pool_out_shape
            flat_pool_out = fluid.layers.flatten(pool_out)
            recombination_out = fluid.layers.fc(input=flat_pool_out,
                                                size=self.new_filters[i] *
                                                self.sparse_feature_dim *
                                                pool_out_shape,
                                                act='tanh')
            new_feature_list.append(recombination_out)
            featuer_generation_input = pool_out
        new_featues = fluid.layers.concat(new_feature_list, axis=1)
        new_features_map = fluid.layers.reshape(
            new_featues,
            shape=[0, new_feature_field_num, self.sparse_feature_dim])
        all_features = fluid.layers.concat(
            [feat_embeddings, new_features_map], axis=1)
        interaction_list = []
        for i in range(all_features.shape[1]):
            for j in range(i + 1, all_features.shape[1]):
                interaction_list.append(
                    fluid.layers.reduce_sum(
                        all_features[:, i, :] * all_features[:, j, :],
                        dim=1,
                        keep_dim=True))
        new_feature_dnn_input = fluid.layers.concat(interaction_list, axis=1)
        feat_embeddings_dnn_input = fluid.layers.reshape(
            feat_embeddings,
            shape=[0, self.num_field * self.sparse_feature_dim])
        dnn_input = fluid.layers.concat(
            [feat_embeddings_dnn_input, new_feature_dnn_input], axis=1)

        # ------------------------- DNN --------------------------

        fcs = [dnn_input]

        for size in self.hidden_layers:
            output = fluid.layers.fc(
                input=fcs[-1],
                size=size,
                act=self.act,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Normal(
                        scale=1.0 / math.sqrt(fcs[-1].shape[1]))))
            fcs.append(output)

        predict = fluid.layers.fc(
            input=fcs[-1],
            size=1,
            act="sigmoid",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(fcs[-1].shape[1]))))

        # ------------------------- Predict --------------------------

        self.predict = predict

        cost = fluid.layers.log_loss(
            input=self.predict, label=fluid.layers.cast(self.label, "float32"))
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
