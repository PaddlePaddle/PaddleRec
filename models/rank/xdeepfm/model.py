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

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase
import paddle


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
        self.layer_sizes_cin = envs.get_global_env(
            "hyper_parameters.layer_sizes_cin", None)
        self.layer_sizes_dnn = envs.get_global_env(
            "hyper_parameters.layer_sizes_dnn", None)
        self.act = envs.get_global_env("hyper_parameters.act", None)

    def net(self, inputs, is_infer=False):
        raw_feat_idx = self._sparse_data_var[1]
        raw_feat_value = self._dense_data_var[0]
        self.label = self._sparse_data_var[0]

        init_value_ = 0.1
        initer = paddle.fluid.initializer.TruncatedNormalInitializer(
            loc=0.0, scale=init_value_)

        is_distributed = True if envs.get_trainer() == "CtrTrainer" else False

        # ------------------------- network input --------------------------

        feat_idx = raw_feat_idx
        feat_value = paddle.fluid.layers.nn.reshape(
            raw_feat_value, [-1, self.num_field, 1])  # None * num_field * 1

        feat_embeddings = paddle.static.nn.embedding(
            input=feat_idx,
            is_sparse=True,
            dtype='float32',
            size=[self.sparse_feature_number + 1, self.sparse_feature_dim],
            padding_idx=0,
            param_attr=paddle.ParamAttr(initializer=initer))
        feat_embeddings = paddle.fluid.layers.nn.reshape(feat_embeddings, [
            -1, self.num_field, self.sparse_feature_dim
        ])  # None * num_field * embedding_size
        # None * num_field * embedding_size
        feat_embeddings = feat_embeddings * feat_value

        # -------------------- linear  --------------------

        weights_linear = paddle.static.nn.embedding(
            input=feat_idx,
            is_sparse=True,
            dtype='float32',
            size=[self.sparse_feature_number + 1, 1],
            padding_idx=0,
            param_attr=paddle.ParamAttr(initializer=initer))
        weights_linear = paddle.fluid.layers.nn.reshape(
            weights_linear, [-1, self.num_field, 1])  # None * num_field * 1
        b_linear = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0))
        y_linear = paddle.sum(x=(weights_linear * feat_value),
                              axis=1) + b_linear

        # -------------------- CIN  --------------------

        Xs = [feat_embeddings]
        last_s = self.num_field
        for s in self.layer_sizes_cin:
            # calculate Z^(k+1) with X^k and X^0
            X_0 = paddle.fluid.layers.nn.reshape(
                paddle.transpose(Xs[0], [0, 2, 1]),
                [-1, self.sparse_feature_dim, self.num_field,
                 1])  # None, embedding_size, num_field, 1
            X_k = paddle.fluid.layers.nn.reshape(
                paddle.transpose(Xs[-1], [0, 2, 1]),
                [-1, self.sparse_feature_dim, 1,
                 last_s])  # None, embedding_size, 1, last_s
            Z_k_1 = paddle.fluid.layers.matmul(
                X_0, X_k)  # None, embedding_size, num_field, last_s

            # compresses Z^(k+1) to X^(k+1)
            Z_k_1 = paddle.fluid.layers.nn.reshape(Z_k_1, [
                -1, self.sparse_feature_dim, last_s * self.num_field
            ])  # None, embedding_size, last_s*num_field
            Z_k_1 = paddle.transpose(
                Z_k_1, [0, 2, 1])  # None, s*num_field, embedding_size
            Z_k_1 = paddle.fluid.layers.nn.reshape(
                Z_k_1,
                [-1, last_s * self.num_field, 1, self.sparse_feature_dim]
            )  # None, last_s*num_field, 1, embedding_size  (None, channal_in, h, w)
            X_k_1 = paddle.fluid.layers.nn.conv2d(
                Z_k_1,
                num_filters=s,
                filter_size=(1, 1),
                act=None,
                bias_attr=False,
                param_attr=paddle.ParamAttr(
                    initializer=initer))  # None, s, 1, embedding_size
            X_k_1 = paddle.fluid.layers.nn.reshape(
                X_k_1,
                [-1, s, self.sparse_feature_dim])  # None, s, embedding_size

            Xs.append(X_k_1)
            last_s = s

        # sum pooling
        y_cin = paddle.concat(
            x=Xs[1:], axis=1)  # None, (num_field++), embedding_size
        y_cin = paddle.sum(x=y_cin, axis=-1)  # None, (num_field++)
        y_cin = paddle.static.nn.fc(
            x=y_cin,
            size=1,
            activation=None,
            weight_attr=paddle.ParamAttr(initializer=initer),
            bias_attr=None)
        y_cin = paddle.sum(x=y_cin, axis=-1, keepdim=True)

        # -------------------- DNN --------------------

        y_dnn = paddle.fluid.layers.nn.reshape(
            feat_embeddings, [-1, self.num_field * self.sparse_feature_dim])
        for s in self.layer_sizes_dnn:
            y_dnn = paddle.static.nn.fc(
                x=y_dnn,
                size=s,
                activation=self.act,
                weight_attr=paddle.ParamAttr(initializer=initer),
                bias_attr=None)
        y_dnn = paddle.static.nn.fc(
            x=y_dnn,
            size=1,
            activation=None,
            weight_attr=paddle.ParamAttr(initializer=initer),
            bias_attr=None)

        # ------------------- xDeepFM ------------------

        self.predict = paddle.nn.functional.sigmoid(y_linear + y_cin + y_dnn)
        cost = paddle.nn.functional.log_loss(
            input=self.predict,
            label=paddle.cast(self.label, "float32"),
            epsilon=0.0000001)
        batch_cost = paddle.mean(x=cost)
        self._cost = batch_cost

        # for auc
        predict_2d = paddle.concat(x=[1 - self.predict, self.predict], axis=1)
        label_int = paddle.cast(self.label, 'int64')
        auc_var, batch_auc_var, _ = paddle.fluid.layers.auc(input=predict_2d,
                                                            label=label_int,
                                                            slide_steps=0)
        self._metrics["AUC"] = auc_var
        self._metrics["BATCH_AUC"] = batch_auc_var
        if is_infer:
            self._infer_results["AUC"] = auc_var
