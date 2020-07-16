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

import paddle.fluid as fluid
import itertools
from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.is_distributed = True if envs.get_fleet_mode().upper(
        ) == "PSLIB" else False
        self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim")
        self.learning_rate = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate")

    def _SENETLayer(self, inputs, filed_size, reduction_ratio=3):
        reduction_size = max(1, filed_size // reduction_ratio)
        Z = fluid.layers.reduce_mean(inputs, dim=-1)

        A_1 = fluid.layers.fc(
            input=Z,
            size=reduction_size,
            param_attr=fluid.initializer.Xavier(uniform=False),
            act='relu',
            name='W_1')

        A_2 = fluid.layers.fc(
            input=A_1,
            size=filed_size,
            param_attr=fluid.initializer.Xavier(uniform=False),
            act='relu',
            name='W_2')

        V = fluid.layers.elementwise_mul(
            inputs, y=fluid.layers.unsqueeze(
                input=A_2, axes=[2]))

        return fluid.layers.split(V, num_or_sections=filed_size, dim=1)

    def _BilinearInteraction(self,
                             inputs,
                             filed_size,
                             embedding_size,
                             bilinear_type="interaction"):
        if bilinear_type == "all":
            p = [
                fluid.layers.elementwise_mul(
                    fluid.layers.fc(
                        input=v_i,
                        size=embedding_size,
                        param_attr=fluid.initializer.Xavier(uniform=False),
                        act=None,
                        name=None),
                    fluid.layers.squeeze(
                        input=v_j, axes=[1]))
                for v_i, v_j in itertools.combinations(inputs, 2)
            ]
        else:
            raise NotImplementedError

        return fluid.layers.concat(input=p, axis=1)

    def _DNNLayer(self, inputs, dropout_rate=0.5):
        deep_input = inputs
        for i, hidden_unit in enumerate([400, 400, 400]):
            fc_out = fluid.layers.fc(
                input=deep_input,
                size=hidden_unit,
                param_attr=fluid.initializer.Xavier(uniform=False),
                act='relu',
                name='d_' + str(i))
            fc_out = fluid.layers.dropout(fc_out, dropout_prob=dropout_rate)
            deep_input = fc_out

        return deep_input

    def net(self, input, is_infer=False):
        self.sparse_inputs = self._sparse_data_var[1:]
        self.dense_input = self._dense_data_var[0]
        self.label_input = self._sparse_data_var[0]

        emb = []
        for data in self.sparse_inputs:
            feat_emb = fluid.embedding(
                input=data,
                size=[self.sparse_feature_number, self.sparse_feature_dim],
                param_attr=fluid.ParamAttr(
                    name='dis_emb',
                    learning_rate=5,
                    initializer=fluid.initializer.Xavier(
                        fan_in=self.sparse_feature_dim,
                        fan_out=self.sparse_feature_dim)),
                is_sparse=True)
            emb.append(feat_emb)
        concat_emb = fluid.layers.concat(emb, axis=1)

        filed_size = len(self.sparse_inputs)
        bilinear_type = envs.get_global_env("hyper_parameters.bilinear_type")
        reduction_ratio = envs.get_global_env(
            "hyper_parameters.reduction_ratio")
        dropout_rate = envs.get_global_env("hyper_parameters.dropout_rate")

        senet_output = self._SENETLayer(concat_emb, filed_size,
                                        reduction_ratio)
        senet_bilinear_out = self._BilinearInteraction(
            senet_output, filed_size, self.sparse_feature_dim, bilinear_type)

        concat_emb = fluid.layers.split(
            concat_emb, num_or_sections=filed_size, dim=1)
        bilinear_out = self._BilinearInteraction(
            concat_emb, filed_size, self.sparse_feature_dim, bilinear_type)
        dnn_input = fluid.layers.concat(
            input=[senet_bilinear_out, bilinear_out, self.dense_input], axis=1)
        dnn_output = self._DNNLayer(dnn_input, dropout_rate)

        y_pred = fluid.layers.fc(
            input=dnn_output,
            size=1,
            param_attr=fluid.initializer.Xavier(uniform=False),
            act='sigmoid',
            name='logit')
        self.predict = y_pred
        auc, batch_auc, _ = fluid.layers.auc(input=self.predict,
                                             label=self.label_input,
                                             num_thresholds=2**12,
                                             slide_steps=20)

        if is_infer:
            self._infer_results["AUC"] = auc
            self._infer_results["BATCH_AUC"] = batch_auc
            return

        self._metrics["AUC"] = auc
        self._metrics["BATCH_AUC"] = batch_auc
        cost = fluid.layers.log_loss(
            input=self.predict,
            label=fluid.layers.cast(
                x=self.label_input, dtype='float32'))
        avg_cost = fluid.layers.reduce_mean(cost)
        self._cost = avg_cost
