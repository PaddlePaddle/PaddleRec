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

import paddle
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
        self.deep_input_size = envs.get_global_env(
            "hyper_parameters.deep_input_size", 50)
        self.use_inner_product = envs.get_global_env(
            "hyper_parameters.use_inner_product", None)
        self.layer_sizes = envs.get_global_env("hyper_parameters.fc_sizes",
                                               None)
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
        # ------------------------- Embedding --------------------------

        feat_embeddings_re = fluid.embedding(
            input=feat_idx,
            is_sparse=True,
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

        # ------------------------- Linear Signal --------------------------

        linear_input_size = self.num_field * self.sparse_feature_dim
        flaten_feat_embedding = fluid.layers.reshape(
            x=feat_embeddings, shape=[-1, linear_input_size])

        w_z_linear_weights = fluid.layers.create_parameter(
            shape=[linear_input_size, self.deep_input_size], dtype="float32")
        linear_signal = fluid.layers.matmul(
            flaten_feat_embedding,
            w_z_linear_weights)  # batch_size * deep_input_size

        # ------------------------- Quardatic Singal --------------------------

        quadratic_output = []
        if self.use_inner_product:
            w_p_quardatic_weights = fluid.layers.create_parameter(
                shape=[self.deep_input_size, self.num_field], dtype="float32")
            for i in range(self.deep_input_size):
                transpose_embedding = fluid.layers.transpose(
                    feat_embeddings, perm=[0, 2, 1])
                theta = fluid.layers.elementwise_mul(
                    transpose_embedding, w_p_quardatic_weights[i], axis=-1)
                quadratic_output.append(
                    paddle.norm(
                        fluid.layers.reduce_sum(
                            theta, dim=1),
                        p=2,
                        axis=1,
                        keepdim=True))
        else:
            w_p_quardatic_weights_outer = fluid.layers.create_parameter(
                shape=[
                    self.deep_input_size, self.sparse_feature_dim,
                    self.sparse_feature_dim
                ],
                dtype="float32")
            embedding_sum = fluid.layers.reduce_sum(feat_embeddings, dim=1)
            p = fluid.layers.matmul(
                fluid.layers.reshape(
                    embedding_sum, shape=[0, -1, 1]),
                fluid.layers.reshape(
                    embedding_sum, shape=[0, 1, -1]))
            for i in range(self.deep_input_size):
                theta = fluid.layers.elementwise_mul(
                    p, w_p_quardatic_weights_outer[i, :, :], axis=-1)
                quadratic_output.append(
                    fluid.layers.reshape(
                        fluid.layers.reduce_sum(
                            theta, dim=[1, 2]),
                        shape=[-1, 1]))

        quadratic_signal = fluid.layers.concat(quadratic_output, axis=1)

        y_dnn = linear_signal + quadratic_signal

        y_dnn = fluid.layers.relu6(y_dnn, threshold=10000000.0)

        for s in self.layer_sizes:
            y_dnn = fluid.layers.fc(
                input=y_dnn,
                size=s,
                act=self.act,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_)),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_)))

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
