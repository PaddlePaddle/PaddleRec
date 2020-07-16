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
from paddlerec.core.model import ModelBase


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
        self.d_model = envs.get_global_env("hyper_parameters.d_model", None)
        self.d_key = envs.get_global_env("hyper_parameters.d_key", None)
        self.d_value = envs.get_global_env("hyper_parameters.d_value", None)
        self.n_head = envs.get_global_env("hyper_parameters.n_head", None)
        self.dropout_rate = envs.get_global_env(
            "hyper_parameters.dropout_rate", 0)
        self.n_interacting_layers = envs.get_global_env(
            "hyper_parameters.n_interacting_layers", 1)

    def multi_head_attention(self, queries, keys, values, d_key, d_value,
                             d_model, n_head, dropout_rate):
        keys = queries if keys is None else keys
        values = keys if values is None else values
        if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3
                ):
            raise ValueError(
                "Inputs: quries, keys and values should all be 3-D tensors.")

        def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
            """
            Add linear projection to queries, keys, and values.
            """
            q = fluid.layers.fc(input=queries,
                                size=d_key * n_head,
                                bias_attr=False,
                                num_flatten_dims=2)
            k = fluid.layers.fc(input=keys,
                                size=d_key * n_head,
                                bias_attr=False,
                                num_flatten_dims=2)
            v = fluid.layers.fc(input=values,
                                size=d_value * n_head,
                                bias_attr=False,
                                num_flatten_dims=2)
            return q, k, v

        def __split_heads_qkv(queries, keys, values, n_head, d_key, d_value):
            """
            Reshape input tensors at the last dimension to split multi-heads 
            and then transpose. Specifically, transform the input tensor with shape
            [bs, max_sequence_length, n_head * hidden_dim] to the output tensor
            with shape [bs, n_head, max_sequence_length, hidden_dim].
            """
            # The value 0 in shape attr means copying the corresponding dimension
            # size of the input as the output dimension size.
            reshaped_q = fluid.layers.reshape(
                x=queries, shape=[0, 0, n_head, d_key], inplace=True)
            # permuate the dimensions into:
            # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
            q = fluid.layers.transpose(x=reshaped_q, perm=[0, 2, 1, 3])
            # For encoder-decoder attention in inference, insert the ops and vars
            # into global block to use as cache among beam search.
            reshaped_k = fluid.layers.reshape(
                x=keys, shape=[0, 0, n_head, d_key], inplace=True)
            k = fluid.layers.transpose(x=reshaped_k, perm=[0, 2, 1, 3])
            reshaped_v = fluid.layers.reshape(
                x=values, shape=[0, 0, n_head, d_value], inplace=True)
            v = fluid.layers.transpose(x=reshaped_v, perm=[0, 2, 1, 3])

            return q, k, v

        def scaled_dot_product_attention(q, k, v, d_key, dropout_rate):
            """
            Scaled Dot-Product Attention
            """
            product = fluid.layers.matmul(
                x=q, y=k, transpose_y=True, alpha=d_key**-0.5)

            weights = fluid.layers.softmax(product)
            if dropout_rate:
                weights = fluid.layers.dropout(
                    weights,
                    dropout_prob=dropout_rate,
                    seed=None,
                    is_test=False)
            out = fluid.layers.matmul(weights, v)
            return out

        def __combine_heads(x):
            """
            Transpose and then reshape the last two dimensions of inpunt tensor x
            so that it becomes one dimension, which is reverse to __split_heads.
            """
            if len(x.shape) != 4:
                raise ValueError("Input(x) should be a 4-D Tensor.")

            trans_x = fluid.layers.transpose(x, perm=[0, 2, 1, 3])
            # The value 0 in shape attr means copying the corresponding dimension
            # size of the input as the output dimension size.
            return fluid.layers.reshape(
                x=trans_x,
                shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
                inplace=True)

        q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)
        q, k, v = __split_heads_qkv(q, k, v, n_head, d_key, d_value)

        ctx_multiheads = scaled_dot_product_attention(q, k, v, self.d_model,
                                                      dropout_rate)

        out = __combine_heads(ctx_multiheads)

        return out

    def interacting_layer(self, x):
        attention_out = self.multi_head_attention(
            x, None, None, self.d_key, self.d_value, self.d_model, self.n_head,
            self.dropout_rate)
        W_0_x = fluid.layers.fc(input=x,
                                size=self.d_model,
                                bias_attr=False,
                                num_flatten_dims=2)
        res_out = fluid.layers.relu(attention_out + W_0_x)

        return res_out

    def net(self, inputs, is_infer=False):
        init_value_ = 0.1
        is_distributed = True if envs.get_trainer() == "CtrTrainer" else False

        # ------------------------- network input --------------------------

        raw_feat_idx = self._sparse_data_var[1]
        raw_feat_value = self._dense_data_var[0]
        self.label = self._sparse_data_var[0]

        feat_idx = raw_feat_idx
        feat_value = fluid.layers.reshape(
            raw_feat_value, [-1, self.num_field, 1])  # None * num_field * 1

        # ------------------------- Embedding --------------------------

        feat_embeddings_re = fluid.embedding(
            input=feat_idx,
            is_sparse=True,
            is_distributed=is_distributed,
            dtype='float32',
            size=[self.sparse_feature_number + 1, self.sparse_feature_dim],
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0,
                    scale=init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))
        feat_embeddings = fluid.layers.reshape(
            feat_embeddings_re,
            shape=[-1, self.num_field, self.sparse_feature_dim
                   ])  # None * num_field * embedding_size
        # None * num_field * embedding_size
        feat_embeddings = feat_embeddings * feat_value

        inter_input = feat_embeddings

        # ------------------------- interacting layer --------------------------

        for _ in range(self.n_interacting_layers):
            interacting_layer_out = self.interacting_layer(inter_input)
            inter_input = interacting_layer_out

        # ------------------------- DNN --------------------------

        dnn_input = fluid.layers.flatten(interacting_layer_out, axis=1)

        y_dnn = fluid.layers.fc(
            input=dnn_input,
            size=1,
            act=None,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=init_value_)))

        self.predict = fluid.layers.sigmoid(y_dnn)
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
