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

from collections import OrderedDict

import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import Model as ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def init_network(self):
        self.cross_num = envs.get_global_env("hyper_parameters.cross_num", None, self._namespace)
        self.dnn_hidden_units = envs.get_global_env("hyper_parameters.dnn_hidden_units", None, self._namespace)
        self.l2_reg_cross = envs.get_global_env("hyper_parameters.l2_reg_cross", None, self._namespace)
        self.dnn_use_bn = envs.get_global_env("hyper_parameters.dnn_use_bn", None, self._namespace)
        self.clip_by_norm = envs.get_global_env("hyper_parameters.clip_by_norm", None, self._namespace)
        cat_feat_num = envs.get_global_env("hyper_parameters.cat_feat_num", None, self._namespace)

        self.sparse_inputs = self._sparse_data_var[1:]
        self.dense_inputs = self._dense_data_var
        self.target_input = self._sparse_data_var[0]

        cat_feat_dims_dict = OrderedDict()
        for line in open(cat_feat_num):
            spls = line.strip().split()
            assert len(spls) == 2
            cat_feat_dims_dict[spls[0]] = int(spls[1])
        self.cat_feat_dims_dict = cat_feat_dims_dict if cat_feat_dims_dict else OrderedDict(
        )
        self.is_sparse = envs.get_global_env("hyper_parameters.is_sparse", None, self._namespace)

        self.dense_feat_names = [i.name for i in self.dense_inputs]
        self.sparse_feat_names = [i.name for i in self.sparse_inputs]

        # {feat_name: dims}
        self.feat_dims_dict = OrderedDict(
            [(feat_name, 1) for feat_name in self.dense_feat_names])
        self.feat_dims_dict.update(self.cat_feat_dims_dict)

        self.net_input = None
        self.loss = None
    
    def _create_embedding_input(self):
        # sparse embedding
        sparse_emb_dict = OrderedDict()
        for var in self.sparse_inputs:
            sparse_emb_dict[var.name] = fluid.embedding(input=var,
                size=[self.feat_dims_dict[var.name] + 1,
                6 * int(pow(self.feat_dims_dict[var.name], 0.25))
                ],is_sparse=self.is_sparse)
            
        # combine dense and sparse_emb
        dense_input_list = self.dense_inputs
        sparse_emb_list = list(sparse_emb_dict.values())

        sparse_input = fluid.layers.concat(sparse_emb_list, axis=-1)
        sparse_input = fluid.layers.flatten(sparse_input)

        dense_input = fluid.layers.concat(dense_input_list, axis=-1)
        dense_input = fluid.layers.flatten(dense_input)
        dense_input = fluid.layers.cast(dense_input, 'float32')

        net_input = fluid.layers.concat([dense_input, sparse_input], axis=-1)

        return net_input

    def _deep_net(self, input, hidden_units, use_bn=False, is_test=False):
        for units in hidden_units:
            input = fluid.layers.fc(input=input, size=units)
            if use_bn:
                input = fluid.layers.batch_norm(input, is_test=is_test)
            input = fluid.layers.relu(input)
        return input

    def _cross_layer(self, x0, x, prefix):
        input_dim = x0.shape[-1]
        w = fluid.layers.create_parameter(
            [input_dim], dtype='float32', name=prefix + "_w")
        b = fluid.layers.create_parameter(
            [input_dim], dtype='float32', name=prefix + "_b")
        xw = fluid.layers.reduce_sum(x * w, dim=1, keep_dim=True)  # (N, 1)
        return x0 * xw + b + x, w

    def _cross_net(self, input, num_corss_layers):
        x = x0 = input
        l2_reg_cross_list = []
        for i in range(num_corss_layers):
            x, w = self._cross_layer(x0, x, "cross_layer_{}".format(i))
            l2_reg_cross_list.append(self._l2_loss(w))
        l2_reg_cross_loss = fluid.layers.reduce_sum(
            fluid.layers.concat(
                l2_reg_cross_list, axis=-1))
        return x, l2_reg_cross_loss

    def _l2_loss(self, w):
        return fluid.layers.reduce_sum(fluid.layers.square(w))

    def train_net(self):
        self.init_network()
        
        self.net_input = self._create_embedding_input()
        
        deep_out = self._deep_net(self.net_input, self.dnn_hidden_units, self.dnn_use_bn, False)

        cross_out, l2_reg_cross_loss = self._cross_net(self.net_input,
                                                       self.cross_num)

        last_out = fluid.layers.concat([deep_out, cross_out], axis=-1)
        logit = fluid.layers.fc(last_out, 1)

        self.prob = fluid.layers.sigmoid(logit)

        # auc
        prob_2d = fluid.layers.concat([1 - self.prob, self.prob], 1)
        label_int = fluid.layers.cast(self.target_input, 'int64')
        auc_var, batch_auc_var, self.auc_states = fluid.layers.auc(
            input=prob_2d, label=label_int, slide_steps=0)
        self._metrics["AUC"] = auc_var
        self._metrics["BATCH_AUC"] = batch_auc_var
        
        # logloss
        logloss = fluid.layers.log_loss(self.prob, fluid.layers.cast(self.target_input, dtype='float32'))
        self.avg_logloss = fluid.layers.reduce_mean(logloss)

        # reg_coeff * l2_reg_cross
        l2_reg_cross_loss = self.l2_reg_cross * l2_reg_cross_loss
        self.loss = self.avg_logloss + l2_reg_cross_loss
        self._cost = self.loss

    def optimizer(self):
        learning_rate = envs.get_global_env("hyper_parameters.learning_rate", None, self._namespace)
        optimizer = fluid.optimizer.Adam(learning_rate, lazy_mode=True)
        return optimizer

    def infer_net(self, parameter_list):
        self.deepfm_net()
