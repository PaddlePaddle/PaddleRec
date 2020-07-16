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

    def _FieldWiseBiInteraction(self, inputs):
        # MF module
        field_wise_embeds_list = inputs

        field_wise_vectors = [
            fluid.layers.reduce_sum(
                field_i_vectors, dim=1, keep_dim=True)
            for field_i_vectors in field_wise_embeds_list
        ]
        num_fields = len(field_wise_vectors)

        h_mf_list = []
        for emb_left, emb_right in itertools.combinations(field_wise_vectors,
                                                          2):
            embeddings_prod = fluid.layers.elementwise_mul(emb_left, emb_right)

            field_weighted_embedding = fluid.layers.fc(
                input=embeddings_prod,
                size=self.sparse_feature_dim,
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                name='kernel_mf')
            h_mf_list.append(field_weighted_embedding)
        h_mf = fluid.layers.concat(h_mf_list, axis=1)
        h_mf = fluid.layers.reshape(
            x=h_mf, shape=[-1, num_fields, self.sparse_feature_dim])
        h_mf = fluid.layers.reduce_sum(h_mf, dim=1)

        square_of_sum_list = [
            fluid.layers.square(
                fluid.layers.reduce_sum(
                    field_i_vectors, dim=1, keep_dim=True))
            for field_i_vectors in field_wise_embeds_list
        ]

        sum_of_square_list = [
            fluid.layers.reduce_sum(
                fluid.layers.elementwise_mul(field_i_vectors, field_i_vectors),
                dim=1,
                keep_dim=True) for field_i_vectors in field_wise_embeds_list
        ]

        field_fm_list = []
        for square_of_sum, sum_of_square in zip(square_of_sum_list,
                                                sum_of_square_list):
            field_fm = fluid.layers.reshape(
                fluid.layers.elementwise_sub(square_of_sum, sum_of_square),
                shape=[-1, self.sparse_feature_dim])
            field_fm = fluid.layers.fc(
                input=field_fm,
                size=self.sparse_feature_dim,
                param_attr=fluid.initializer.ConstantInitializer(value=0.5),
                name='kernel_fm')
            field_fm_list.append(field_fm)

        h_fm = fluid.layers.concat(field_fm_list, axis=1)
        h_fm = fluid.layers.reshape(
            x=h_fm, shape=[-1, num_fields, self.sparse_feature_dim])
        h_fm = fluid.layers.reduce_sum(h_fm, dim=1)

        return fluid.layers.elementwise_add(h_mf, h_fm)

    def _DNNLayer(self, inputs, dropout_rate=0.2):
        deep_input = inputs
        for i, hidden_unit in enumerate([64, 32]):
            fc_out = fluid.layers.fc(
                input=deep_input,
                size=hidden_unit,
                param_attr=fluid.initializer.Xavier(uniform=False),
                act='relu',
                name='d_' + str(i))
            fc_out = fluid.layers.dropout(fc_out, dropout_prob=dropout_rate)
            deep_input = fc_out

        return deep_input

    def _embeddingLayer(self, inputs):
        emb_list = []
        in_len = len(inputs)
        for data in inputs:
            feat_emb = fluid.embedding(
                input=data,
                size=[self.sparse_feature_number, self.sparse_feature_dim],
                param_attr=fluid.ParamAttr(
                    name='item_emb',
                    learning_rate=5,
                    initializer=fluid.initializer.Xavier(
                        fan_in=self.sparse_feature_dim,
                        fan_out=self.sparse_feature_dim)),
                is_sparse=True)
            emb_list.append(feat_emb)
        concat_emb = fluid.layers.concat(emb_list, axis=1)
        field_emb = fluid.layers.reshape(
            x=concat_emb, shape=[-1, in_len, self.sparse_feature_dim])

        return field_emb

    def net(self, input, is_infer=False):
        self.user_inputs = self._sparse_data_var[1:13]
        self.item_inputs = self._sparse_data_var[13:16]
        self.contex_inputs = self._sparse_data_var[16:]
        self.label_input = self._sparse_data_var[0]

        dropout_rate = envs.get_global_env("hyper_parameters.dropout_rate")

        field_wise_embeds_list = []
        for inputs in [self.user_inputs, self.item_inputs, self.contex_inputs]:
            field_emb = self._embeddingLayer(inputs)
            field_wise_embeds_list.append(field_emb)

        dnn_input = fluid.layers.concat(
            [
                fluid.layers.flatten(
                    x=field_i_vectors, axis=1)
                for field_i_vectors in field_wise_embeds_list
            ],
            axis=1)

        #mlp part
        dnn_output = self._DNNLayer(dnn_input, dropout_rate)

        #field-weighted embedding
        fm_mf_out = self._FieldWiseBiInteraction(field_wise_embeds_list)
        logits = fluid.layers.concat([fm_mf_out, dnn_output], axis=1)

        y_pred = fluid.layers.fc(
            input=logits,
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
