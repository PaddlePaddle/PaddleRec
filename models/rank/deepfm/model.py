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

import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import Model as ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def deepfm_net(self):
        init_value_ = 0.1
        is_distributed = True if envs.get_trainer() == "CtrTrainer" else False
        sparse_feature_number = envs.get_global_env("hyper_parameters.sparse_feature_number", None, self._namespace)
        sparse_feature_dim = envs.get_global_env("hyper_parameters.sparse_feature_dim", None, self._namespace)

        # ------------------------- network input --------------------------

        num_field = envs.get_global_env("hyper_parameters.num_field", None, self._namespace)
        
        raw_feat_idx = self._sparse_data_var[1]
        raw_feat_value = self._dense_data_var[0]
        self.label = self._sparse_data_var[0]
        
        feat_idx = raw_feat_idx
        feat_value = fluid.layers.reshape(raw_feat_value, [-1, num_field, 1])  # None * num_field * 1
       
        reg = envs.get_global_env("hyper_parameters.reg", 1e-4, self._namespace)
        first_weights_re = fluid.embedding(
            input=feat_idx,
            is_sparse=True,
            is_distributed=is_distributed,
            dtype='float32',
            size=[sparse_feature_number + 1, 1],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_),
                regularizer=fluid.regularizer.L1DecayRegularizer(reg)))
        first_weights = fluid.layers.reshape(
            first_weights_re, shape=[-1, num_field, 1])  # None * num_field * 1
        y_first_order = fluid.layers.reduce_sum((first_weights * feat_value), 1)

        # ------------------------- second order term --------------------------

        feat_embeddings_re = fluid.embedding(
            input=feat_idx,
            is_sparse=True,
            is_distributed=is_distributed,
            dtype='float32',
            size=[sparse_feature_number + 1, sparse_feature_dim],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_ / math.sqrt(float(sparse_feature_dim)))))
        feat_embeddings = fluid.layers.reshape(
            feat_embeddings_re,
            shape=[-1, num_field,
                   sparse_feature_dim])  # None * num_field * embedding_size
        feat_embeddings = feat_embeddings * feat_value  # None * num_field * embedding_size

        # sum_square part
        summed_features_emb = fluid.layers.reduce_sum(feat_embeddings,
                                                      1)  # None * embedding_size
        summed_features_emb_square = fluid.layers.square(
            summed_features_emb)  # None * embedding_size

        # square_sum part
        squared_features_emb = fluid.layers.square(
            feat_embeddings)  # None * num_field * embedding_size
        squared_sum_features_emb = fluid.layers.reduce_sum(
            squared_features_emb, 1)  # None * embedding_size

        y_second_order = 0.5 * fluid.layers.reduce_sum(
            summed_features_emb_square - squared_sum_features_emb, 1,
            keep_dim=True)  # None * 1

        # ------------------------- DNN --------------------------

        layer_sizes = envs.get_global_env("hyper_parameters.fc_sizes", None, self._namespace)
        act = envs.get_global_env("hyper_parameters.act", None, self._namespace)
        y_dnn = fluid.layers.reshape(feat_embeddings,
                                     [-1, num_field * sparse_feature_dim])
        for s in layer_sizes:
            y_dnn = fluid.layers.fc(
                input=y_dnn,
                size=s,
                act=act,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_ / math.sqrt(float(10)))),
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

        # ------------------------- DeepFM --------------------------

        self.predict = fluid.layers.sigmoid(y_first_order + y_second_order + y_dnn)

    def train_net(self):
        self.deepfm_net()

        # ------------------------- Cost(logloss) --------------------------

        cost = fluid.layers.log_loss(input=self.predict, label=fluid.layers.cast(self.label, "float32"))
        avg_cost = fluid.layers.reduce_sum(cost)

        self._cost = avg_cost

        # ------------------------- Metric(Auc) --------------------------

        predict_2d = fluid.layers.concat([1 - self.predict, self.predict], 1)
        label_int = fluid.layers.cast(self.label, 'int64')
        auc_var, batch_auc_var, _ = fluid.layers.auc(input=predict_2d,
                                                     label=label_int,
                                                     slide_steps=0)
        self._metrics["AUC"] = auc_var
        self._metrics["BATCH_AUC"] = batch_auc_var

    def optimizer(self):
        learning_rate = envs.get_global_env("hyper_parameters.learning_rate", None, self._namespace)
        optimizer = fluid.optimizer.Adam(learning_rate, lazy_mode=True)
        return optimizer

    def infer_net(self, parameter_list):
        self.deepfm_net()
