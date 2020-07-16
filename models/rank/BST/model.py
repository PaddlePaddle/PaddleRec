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
from functools import partial

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


def positionwise_feed_forward(x, d_inner_hid, d_hid, dropout_rate):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act="relu")
    if dropout_rate:
        hidden = layers.dropout(
            hidden,
            dropout_prob=dropout_rate,
            seed=dropout_seed,
            is_test=False)
    out = layers.fc(input=hidden, size=d_hid, num_flatten_dims=2)
    return out


def pre_post_process_layer(prev_out, out, process_cmd, dropout_rate=0.):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out = layers.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.initializer.Constant(1.),
                bias_attr=fluid.initializer.Constant(0.))
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    seed=dropout_seed,
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.item_emb_size = envs.get_global_env(
            "hyper_parameters.item_emb_size", 64)
        self.cat_emb_size = envs.get_global_env(
            "hyper_parameters.cat_emb_size", 64)
        self.position_emb_size = envs.get_global_env(
            "hyper_parameters.position_emb_size", 64)
        self.act = envs.get_global_env("hyper_parameters.act", "sigmoid")
        self.is_sparse = envs.get_global_env("hyper_parameters.is_sparse",
                                             False)
        # significant for speeding up the training process
        self.use_DataLoader = envs.get_global_env(
            "hyper_parameters.use_DataLoader", False)
        self.item_count = envs.get_global_env("hyper_parameters.item_count",
                                              63001)
        self.cat_count = envs.get_global_env("hyper_parameters.cat_count", 801)
        self.position_count = envs.get_global_env(
            "hyper_parameters.position_count", 5001)
        self.n_encoder_layers = envs.get_global_env(
            "hyper_parameters.n_encoder_layers", 1)
        self.d_model = envs.get_global_env("hyper_parameters.d_model", 96)
        self.d_key = envs.get_global_env("hyper_parameters.d_key", None)
        self.d_value = envs.get_global_env("hyper_parameters.d_value", None)
        self.n_head = envs.get_global_env("hyper_parameters.n_head", None)
        self.dropout_rate = envs.get_global_env(
            "hyper_parameters.dropout_rate", 0.0)
        self.postprocess_cmd = envs.get_global_env(
            "hyper_parameters.postprocess_cmd", "da")
        self.preprocess_cmd = envs.get_global_env(
            "hyper_parameters.postprocess_cmd", "n")
        self.prepostprocess_dropout = envs.get_global_env(
            "hyper_parameters.prepostprocess_dropout", 0.0)
        self.d_inner_hid = envs.get_global_env("hyper_parameters.d_inner_hid",
                                               512)
        self.relu_dropout = envs.get_global_env(
            "hyper_parameters.relu_dropout", 0.0)
        self.layer_sizes = envs.get_global_env("hyper_parameters.fc_sizes",
                                               None)

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

        ctx_multiheads = scaled_dot_product_attention(q, k, v, d_model,
                                                      dropout_rate)

        out = __combine_heads(ctx_multiheads)

        proj_out = fluid.layers.fc(input=out,
                                   size=d_model,
                                   bias_attr=False,
                                   num_flatten_dims=2)

        return proj_out

    def encoder_layer(self, x):
        attention_out = self.multi_head_attention(
            pre_process_layer(x, self.preprocess_cmd,
                              self.prepostprocess_dropout), None, None,
            self.d_key, self.d_value, self.d_model, self.n_head,
            self.dropout_rate)
        attn_output = post_process_layer(x, attention_out,
                                         self.postprocess_cmd,
                                         self.prepostprocess_dropout)
        ffd_output = positionwise_feed_forward(
            pre_process_layer(attn_output, self.preprocess_cmd,
                              self.prepostprocess_dropout), self.d_inner_hid,
            self.d_model, self.relu_dropout)
        return post_process_layer(attn_output, ffd_output,
                                  self.postprocess_cmd,
                                  self.prepostprocess_dropout)

    def net(self, inputs, is_infer=False):

        init_value_ = 0.1

        hist_item_seq = self._sparse_data_var[1]
        hist_cat_seq = self._sparse_data_var[2]
        position_seq = self._sparse_data_var[3]
        target_item = self._sparse_data_var[4]
        target_cat = self._sparse_data_var[5]
        target_position = self._sparse_data_var[6]
        self.label = self._sparse_data_var[0]

        item_emb_attr = fluid.ParamAttr(name="item_emb")
        cat_emb_attr = fluid.ParamAttr(name="cat_emb")
        position_emb_attr = fluid.ParamAttr(name="position_emb")

        hist_item_emb = fluid.embedding(
            input=hist_item_seq,
            size=[self.item_count, self.item_emb_size],
            param_attr=item_emb_attr,
            is_sparse=self.is_sparse)

        hist_cat_emb = fluid.embedding(
            input=hist_cat_seq,
            size=[self.cat_count, self.cat_emb_size],
            param_attr=cat_emb_attr,
            is_sparse=self.is_sparse)

        hist_position_emb = fluid.embedding(
            input=hist_cat_seq,
            size=[self.position_count, self.position_emb_size],
            param_attr=position_emb_attr,
            is_sparse=self.is_sparse)

        target_item_emb = fluid.embedding(
            input=target_item,
            size=[self.item_count, self.item_emb_size],
            param_attr=item_emb_attr,
            is_sparse=self.is_sparse)

        target_cat_emb = fluid.embedding(
            input=target_cat,
            size=[self.cat_count, self.cat_emb_size],
            param_attr=cat_emb_attr,
            is_sparse=self.is_sparse)

        target_position_emb = fluid.embedding(
            input=target_position,
            size=[self.position_count, self.position_emb_size],
            param_attr=position_emb_attr,
            is_sparse=self.is_sparse)

        item_sequence_target = fluid.layers.reduce_sum(
            fluid.layers.sequence_concat([hist_item_emb, target_item_emb]),
            dim=1)
        cat_sequence_target = fluid.layers.reduce_sum(
            fluid.layers.sequence_concat([hist_cat_emb, target_cat_emb]),
            dim=1)
        position_sequence_target = fluid.layers.reduce_sum(
            fluid.layers.sequence_concat(
                [hist_position_emb, target_position_emb]),
            dim=1)

        whole_embedding_withlod = fluid.layers.concat(
            [
                item_sequence_target, cat_sequence_target,
                position_sequence_target
            ],
            axis=1)
        pad_value = fluid.layers.assign(input=np.array(
            [0.0], dtype=np.float32))
        whole_embedding, _ = fluid.layers.sequence_pad(whole_embedding_withlod,
                                                       pad_value)

        for _ in range(self.n_encoder_layers):
            enc_output = self.encoder_layer(whole_embedding)
            enc_input = enc_output
        enc_output = pre_process_layer(enc_output, self.preprocess_cmd,
                                       self.prepostprocess_dropout)

        dnn_input = fluid.layers.reduce_sum(enc_output, dim=1)

        for s in self.layer_sizes:
            dnn_input = fluid.layers.fc(
                input=dnn_input,
                size=s,
                act=self.act,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_ / math.sqrt(float(10)))),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_)))

        y_dnn = fluid.layers.fc(input=dnn_input, size=1, act=None)

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
