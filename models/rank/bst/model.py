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

import paddle
import math
from functools import partial

import numpy as np


class BSTLayer(paddle.nn.Layer):
    def __init__(self, item_emb_size, cat_emb_size, position_emb_size, act,
                 is_sparse, use_DataLoader, item_count, cat_count,
                 position_count, n_encoder_layers, d_model, d_key, d_value,
                 n_head, dropout_rate, postprocess_cmd, preprocess_cmd,
                 prepostprocess_dropout, d_inner_hid, relu_dropout,
                 layer_sizes):
        super(BSTLayer, self).__init__()

        self.item_emb_size = item_emb_size
        self.cat_emb_size = cat_emb_size
        self.position_emb_size = position_emb_size
        self.act = act
        self.is_sparse = is_sparse
        # significant for speeding up the training process
        self.use_DataLoader = use_DataLoader
        self.item_count = item_count
        self.cat_count = cat_count
        self.position_count = position_count
        self.n_encoder_layers = n_encoder_layers
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.postprocess_cmd = postprocess_cmd
        self.preprocess_cmd = preprocess_cmd
        self.prepostprocess_dropout = prepostprocess_dropout
        self.d_inner_hid = d_inner_hid
        self.relu_dropout = relu_dropout
        self.layer_sizes = layer_sizes

        self.bst = BST(item_emb_size, cat_emb_size, position_emb_size, act,
                       is_sparse, use_DataLoader, item_count, cat_count,
                       position_count, n_encoder_layers, d_model, d_key,
                       d_value, n_head, dropout_rate, postprocess_cmd,
                       preprocess_cmd, prepostprocess_dropout, d_inner_hid,
                       relu_dropout, layer_sizes)

        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, userid, hist_item_seq, hist_cat_seq, position_seq,
                target_item, target_cat, target_position):

        y_bst = self.bst(userid, hist_item_seq, hist_cat_seq, position_seq,
                         target_item, target_cat, target_position)

        predict = paddle.nn.functional.sigmoid(y_bst)
        return predict


def positionwise_feed_forward(x, d_inner_hid, d_hid, dropout_rate):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = paddle.static.nn.fc(x=x,
                                 size=d_inner_hid,
                                 num_flatten_dims=2,
                                 activation="relu")
    if dropout_rate:
        hidden = paddle.fluid.layers.nn.dropout(
            hidden,
            dropout_prob=dropout_rate,
            seed=dropout_seed,
            is_test=False)
    out = paddle.static.nn.fc(x=hidden, size=d_hid, num_flatten_dims=2)
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
            out = paddle.static.nn.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=paddle.nn.initializer.Constant(value=1.),
                bias_attr=paddle.nn.initializer.Constant(value=0.))
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = paddle.fluid.layers.nn.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    seed=dropout_seed,
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


class BST(paddle.nn.Layer):
    def __init__(self, item_emb_size, cat_emb_size, position_emb_size, act,
                 is_sparse, use_DataLoader, item_count, cat_count,
                 position_count, n_encoder_layers, d_model, d_key, d_value,
                 n_head, dropout_rate, postprocess_cmd, preprocess_cmd,
                 prepostprocess_dropout, d_inner_hid, relu_dropout,
                 layer_sizes):

        super(BST, self).__init__()
        self.item_emb_size = item_emb_size
        self.cat_emb_size = cat_emb_size
        self.position_emb_size = position_emb_size
        self.act = act
        self.is_sparse = is_sparse
        # significant for speeding up the training process
        self.use_DataLoader = use_DataLoader
        self.item_count = item_count
        self.cat_count = cat_count
        self.position_count = position_count
        self.n_encoder_layers = n_encoder_layers
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.postprocess_cmd = postprocess_cmd
        self.preprocess_cmd = preprocess_cmd
        self.prepostprocess_dropout = prepostprocess_dropout
        self.d_inner_hid = d_inner_hid
        self.relu_dropout = relu_dropout
        self.layer_sizes = layer_sizes

        init_value_ = 0.1
        self.hist_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.item_emb_size)))))

        self.hist_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.cat_emb_size)))))

        self.hist_position_emb_attr = paddle.nn.Embedding(
            self.position_count,
            self.position_emb_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ /
                    math.sqrt(float(self.position_emb_size)))))

        self.target_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.item_emb_size)))))

        self.target_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.cat_emb_size)))))

        self.target_position_emb_attr = paddle.nn.Embedding(
            self.position_count,
            self.position_emb_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ /
                    math.sqrt(float(self.position_emb_size)))))

    def multi_head_attention(self, queries, keys, values, d_key, d_value,
                             d_model, n_head, dropout_rate):
        keys = queries if keys is None else keys
        values = keys if values is None else values
        #print(keys.shape)
        if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3
                ):
            raise ValueError(
                "Inputs: quries, keys and values should all be 3-D tensors.")

        def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
            """
            Add linear projection to queries, keys, and values.
            """
            '''
            print(queries.shape)
            #queries = paddle.reshape(queries, [-1, d_key])
            q_linear = paddle.nn.Linear(
                in_features = d_key,
                out_features = d_key * n_head,
                weight_attr=paddle.ParamAttr(
                    #regularizer=L2Decay(coeff=0.0001),
                    initializer=paddle.nn.initializer.Normal(
                        std=0.1 / math.sqrt(d_key))))
            self.add_sublayer("q_liner", q_linear)
           
            keys = paddle.reshape(keys, [-1, d_key])
            k_linear = paddle.nn.Linear(
                in_features = d_key,
                out_features = d_key * n_head,
                weight_attr=paddle.ParamAttr(
                    #regularizer=L2Decay(coeff=0.0001),
                    initializer=paddle.nn.initializer.Normal(
                        std=0.1 / math.sqrt(d_key))))
            self.add_sublayer("k_liner", k_linear)

            values = paddle.reshape(values, [-1, d_value])
            v_linear = paddle.nn.Linear(
                in_features = d_value,
                out_features = d_value * n_head,
                weight_attr=paddle.ParamAttr(
                    #regularizer=L2Decay(coeff=0.0001),
                    initializer=paddle.nn.initializer.Normal(
                        std=0.1 / math.sqrt(d_value))))
            self.add_sublayer("v_liner", v_linear)

            q = q_linear(queries)
            k = k_linear(keys) 
            v = v_linear(values)
            return q, k, v
            '''
            return queries, keys, values

        def __split_heads_qkv(queries, keys, values, n_head, d_key, d_value):
            """
            Reshape input tensors at the last dimension to split multi-heads 
            and then transpose. Specifically, transform the input tensor with shape
            [bs, max_sequence_length, n_head * hidden_dim] to the output tensor
            with shape [bs, n_head, max_sequence_length, hidden_dim].
            """
            # The value 0 in shape attr means copying the corresponding dimension
            # size of the input as the output dimension size.
            reshaped_q = paddle.reshape(x=queries, shape=[0, 0, n_head, d_key])
            # permuate the dimensions into:
            # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
            q = paddle.transpose(x=reshaped_q, perm=[0, 2, 1, 3])
            # For encoder-decoder attention in inference, insert the ops and vars
            # into global block to use as cache among beam search.
            reshaped_k = paddle.reshape(x=keys, shape=[0, 0, n_head, d_key])
            k = paddle.transpose(x=reshaped_k, perm=[0, 2, 1, 3])
            reshaped_v = paddle.reshape(
                x=values, shape=[0, 0, n_head, d_value])
            v = paddle.transpose(x=reshaped_v, perm=[0, 2, 1, 3])

            return q, k, v

        def scaled_dot_product_attention(q, k, v, d_key, dropout_rate):
            """
            Scaled Dot-Product Attention
            """
            product = paddle.matmul(x=q, y=k, transpose_y=True)

            weights = paddle.nn.functional.softmax(x=product)
            if dropout_rate:
                weights = paddle.fluid.layers.nn.dropout(
                    weights,
                    dropout_prob=dropout_rate,
                    seed=None,
                    is_test=False)
            out = paddle.matmul(x=weights, y=v)
            return out

        def __combine_heads(x):
            """
            Transpose and then reshape the last two dimensions of inpunt tensor x
            so that it becomes one dimension, which is reverse to __split_heads.
            """
            if len(x.shape) != 4:
                raise ValueError("Input(x) should be a 4-D Tensor.")

            trans_x = paddle.transpose(x, perm=[0, 2, 1, 3])
            # The value 0 in shape attr means copying the corresponding dimension
            # size of the input as the output dimension size.
            return paddle.reshape(
                x=trans_x, shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]])

        q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)
        q, k, v = __split_heads_qkv(q, k, v, n_head, d_key, d_value)

        ctx_multiheads = scaled_dot_product_attention(q, k, v, d_model,
                                                      dropout_rate)

        out = __combine_heads(ctx_multiheads)

        proj_out = paddle.static.nn.fc(x=out,
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

    def forward(self, userid, hist_item_seq, hist_cat_seq, position_seq,
                target_item, target_cat, target_position):

        init_value_ = 0.1
        hist_item_emb = self.hist_item_emb_attr(hist_item_seq)

        hist_cat_emb = self.hist_cat_emb_attr(hist_cat_seq)

        hist_position_emb = self.hist_position_emb_attr(position_seq)

        target_item_emb = self.target_item_emb_attr(target_item)

        target_cat_emb = self.target_cat_emb_attr(target_cat)

        target_position_emb = self.target_position_emb_attr(target_position)

        item_sequence_target = paddle.concat(
            [hist_item_emb, target_item_emb], axis=1)

        cat_sequence_target = paddle.concat(
            [hist_cat_emb, target_cat_emb], axis=1)

        position_sequence_target = paddle.concat(
            [hist_position_emb, target_position_emb], axis=1)
        #print(position_sequence_target) 
        whole_embedding_withlod = paddle.concat(
            [
                item_sequence_target, cat_sequence_target,
                position_sequence_target
            ],
            axis=2)
        print(whole_embedding_withlod.shape)
        pad_value = paddle.assign(np.array([0.0]).astype(np.float32))
        #whole_embedding, _ = paddle.nn.functional.pad(whole_embedding_withlod,
        #                                               pad_value=pad_value)

        whole_embedding = whole_embedding_withlod
        for _ in range(self.n_encoder_layers):
            enc_output = self.encoder_layer(whole_embedding)
            enc_input = enc_output

        enc_output = pre_process_layer(enc_output, self.preprocess_cmd,
                                       self.prepostprocess_dropout)

        dnn_input = paddle.sum(x=enc_output, axis=1)

        for s in self.layer_sizes:
            dnn_input = paddle.static.nn.fc(
                x=dnn_input,
                size=s,
                activation=self.act,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.fluid.initializer.
                    TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_ / math.sqrt(float(10)))),
                bias_attr=paddle.ParamAttr(
                    initializer=paddle.fluid.initializer.
                    TruncatedNormalInitializer(
                        loc=0.0, scale=init_value_)))

        y_dnn = paddle.static.nn.fc(x=dnn_input, size=1, activation=None)

        return y_dnn
