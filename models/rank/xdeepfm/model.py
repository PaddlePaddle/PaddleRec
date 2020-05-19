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

import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import Model as ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def xdeepfm_net(self):
        init_value_ = 0.1
        initer = fluid.initializer.TruncatedNormalInitializer(
            loc=0.0, scale=init_value_)
        
        is_distributed = True if envs.get_trainer() == "CtrTrainer" else False
        sparse_feature_number = envs.get_global_env("hyper_parameters.sparse_feature_number", None, self._namespace)
        sparse_feature_dim = envs.get_global_env("hyper_parameters.sparse_feature_dim", None, self._namespace)
        
        # ------------------------- network input --------------------------
        
        num_field = envs.get_global_env("hyper_parameters.num_field", None, self._namespace)
        raw_feat_idx = fluid.data(name='feat_idx', shape=[None, num_field], dtype='int64')
        raw_feat_value = fluid.data(name='feat_value', shape=[None, num_field], dtype='float32')
        self.label = fluid.data(name='label', shape=[None, 1], dtype='float32')  # None * 1
        feat_idx = fluid.layers.reshape(raw_feat_idx, [-1, 1])  # (None * num_field) * 1
        feat_value = fluid.layers.reshape(raw_feat_value, [-1, num_field, 1])  # None * num_field * 1

        feat_embeddings = fluid.embedding(
            input=feat_idx,
            is_sparse=True,
            dtype='float32',
            size=[sparse_feature_number + 1, sparse_feature_dim],
            padding_idx=0,
            param_attr=fluid.ParamAttr(initializer=initer))
        feat_embeddings = fluid.layers.reshape(
            feat_embeddings,
            [-1, num_field, sparse_feature_dim])  # None * num_field * embedding_size
        feat_embeddings = feat_embeddings * feat_value  # None * num_field * embedding_size
        
        # ------------------------- set _data_var --------------------------
        
        self._data_var.append(raw_feat_idx)
        self._data_var.append(raw_feat_value)
        self._data_var.append(self.label)
        if self._platform != "LINUX":
            self._data_loader = fluid.io.DataLoader.from_generator(
                feed_list=self._data_var, capacity=64, use_double_buffer=False, iterable=False)
        
        # -------------------- linear  --------------------

        weights_linear = fluid.embedding(
            input=feat_idx,
            is_sparse=True,
            dtype='float32',
            size=[sparse_feature_number + 1, 1],
            padding_idx=0,
            param_attr=fluid.ParamAttr(initializer=initer))
        weights_linear = fluid.layers.reshape(
            weights_linear, [-1, num_field, 1])  # None * num_field * 1
        b_linear = fluid.layers.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=fluid.initializer.ConstantInitializer(value=0))
        y_linear = fluid.layers.reduce_sum(
            (weights_linear * feat_value), 1) + b_linear
        
        # -------------------- CIN  --------------------

        layer_sizes_cin = envs.get_global_env("hyper_parameters.layer_sizes_cin", None, self._namespace)
        Xs = [feat_embeddings]
        last_s = num_field
        for s in layer_sizes_cin:
            # calculate Z^(k+1) with X^k and X^0
            X_0 = fluid.layers.reshape(
                fluid.layers.transpose(Xs[0], [0, 2, 1]),
                [-1, sparse_feature_dim, num_field,
                1])  # None, embedding_size, num_field, 1
            X_k = fluid.layers.reshape(
                fluid.layers.transpose(Xs[-1], [0, 2, 1]),
                [-1, sparse_feature_dim, 1, last_s])  # None, embedding_size, 1, last_s
            Z_k_1 = fluid.layers.matmul(
                X_0, X_k)  # None, embedding_size, num_field, last_s

            # compresses Z^(k+1) to X^(k+1)
            Z_k_1 = fluid.layers.reshape(Z_k_1, [
                -1, sparse_feature_dim, last_s * num_field
            ])  # None, embedding_size, last_s*num_field
            Z_k_1 = fluid.layers.transpose(
                Z_k_1, [0, 2, 1])  # None, s*num_field, embedding_size
            Z_k_1 = fluid.layers.reshape(
                Z_k_1, [-1, last_s * num_field, 1, sparse_feature_dim]
            )  # None, last_s*num_field, 1, embedding_size  (None, channal_in, h, w) 
            X_k_1 = fluid.layers.conv2d(
                Z_k_1,
                num_filters=s,
                filter_size=(1, 1),
                act=None,
                bias_attr=False,
                param_attr=fluid.ParamAttr(
                    initializer=initer))  # None, s, 1, embedding_size
            X_k_1 = fluid.layers.reshape(
                X_k_1, [-1, s, sparse_feature_dim])  # None, s, embedding_size

            Xs.append(X_k_1)
            last_s = s

        # sum pooling
        y_cin = fluid.layers.concat(Xs[1:],
                                    1)  # None, (num_field++), embedding_size
        y_cin = fluid.layers.reduce_sum(y_cin, -1)  # None, (num_field++)
        y_cin = fluid.layers.fc(input=y_cin,
                                size=1,
                                act=None,
                                param_attr=fluid.ParamAttr(initializer=initer),
                                bias_attr=None)
        y_cin = fluid.layers.reduce_sum(y_cin, dim=-1, keep_dim=True)

        # -------------------- DNN --------------------

        layer_sizes_dnn = envs.get_global_env("hyper_parameters.layer_sizes_dnn", None, self._namespace)
        act = envs.get_global_env("hyper_parameters.act", None, self._namespace)
        y_dnn = fluid.layers.reshape(feat_embeddings,
                                    [-1, num_field * sparse_feature_dim])
        for s in layer_sizes_dnn:
            y_dnn = fluid.layers.fc(input=y_dnn,
                                    size=s,
                                    act=act,
                                    param_attr=fluid.ParamAttr(initializer=initer),
                                    bias_attr=None)
        y_dnn = fluid.layers.fc(input=y_dnn,
                                size=1,
                                act=None,
                                param_attr=fluid.ParamAttr(initializer=initer),
                                bias_attr=None)

        # ------------------- xDeepFM ------------------

        self.predict = fluid.layers.sigmoid(y_linear + y_cin + y_dnn)
        
    def train_net(self):
        self.xdeepfm_net()

        cost = fluid.layers.log_loss(input=self.predict, label=self.label, epsilon=0.0000001)
        batch_cost = fluid.layers.reduce_mean(cost)
        self._cost = batch_cost

        # for auc
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
        self.xdeepfm_net()