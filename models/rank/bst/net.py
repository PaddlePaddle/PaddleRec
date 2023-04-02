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
    def __init__(self, user_count, item_emb_size, cat_emb_size,
                 position_emb_size, act, is_sparse, use_DataLoader, item_count,
                 cat_count, position_count, n_encoder_layers, d_model, d_key,
                 d_value, n_head, dropout_rate, postprocess_cmd,
                 preprocess_cmd, prepostprocess_dropout, d_inner_hid,
                 relu_dropout, layer_sizes):
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
        self.user_count = user_count
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

        self.bst = BST(user_count, item_emb_size, cat_emb_size,
                       position_emb_size, act, is_sparse, use_DataLoader,
                       item_count, cat_count, position_count, n_encoder_layers,
                       d_model, d_key, d_value, n_head, dropout_rate,
                       postprocess_cmd, preprocess_cmd, prepostprocess_dropout,
                       d_inner_hid, relu_dropout, layer_sizes)

        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, userid, hist_item_seq, hist_cat_seq, position_seq,
                target_item, target_cat, target_position):

        y_bst = self.bst.forward(userid, hist_item_seq, hist_cat_seq,
                                 position_seq, target_item, target_cat,
                                 target_position)

        predict = paddle.nn.functional.sigmoid(y_bst + self.bias)
        return predict


class BST(paddle.nn.Layer):
    def __init__(self, user_count, item_emb_size, cat_emb_size,
                 position_emb_size, act, is_sparse, use_DataLoader, item_count,
                 cat_count, position_count, n_encoder_layers, d_model, d_key,
                 d_value, n_head, dropout_rate, postprocess_cmd,
                 preprocess_cmd, prepostprocess_dropout, d_inner_hid,
                 relu_dropout, layer_sizes):

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
        self.user_count = user_count
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
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.item_emb_size)))))

        self.hist_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.cat_emb_size)))))

        self.hist_position_emb_attr = paddle.nn.Embedding(
            self.position_count,
            self.position_emb_size,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ /
                    math.sqrt(float(self.position_emb_size)))))

        self.target_item_emb_attr = paddle.nn.Embedding(
            self.item_count,
            self.item_emb_size,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.item_emb_size)))))

        self.target_cat_emb_attr = paddle.nn.Embedding(
            self.cat_count,
            self.cat_emb_size,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.cat_emb_size)))))

        self.target_position_emb_attr = paddle.nn.Embedding(
            self.position_count,
            self.position_emb_size,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ /
                    math.sqrt(float(self.position_emb_size)))))

        self.userid_attr = paddle.nn.Embedding(
            self.user_count,
            self.d_model,
            sparse=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=init_value_ / math.sqrt(float(self.d_model)))))

        self._dnn_layers = []
        sizes = [d_model] + layer_sizes + [1]
        acts = ["relu" for _ in range(len(layer_sizes))] + [None]
        for i in range(len(layer_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=0.1 / math.sqrt(sizes[i]))))
            self.add_sublayer('dnn_linear_%d' % i, linear)
            self._dnn_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.LeakyReLU()
                self.add_sublayer('dnn_act_%d' % i, act)
                self._dnn_layers.append(act)

        self.drop_out = paddle.nn.Dropout(p=dropout_rate)

        self.pff_layer = []
        hid_linear = paddle.nn.Linear(
            in_features=self.d_model,
            out_features=self.d_inner_hid,
            weight_attr=paddle.ParamAttr(
                #regularizer=L2Decay(coeff=0.0001),
                initializer=paddle.nn.initializer.Normal(
                    std=0.1 / math.sqrt(self.d_inner_hid))))
        self.add_sublayer('hid_l', hid_linear)
        self.pff_layer.append(hid_linear)

        m = paddle.nn.LeakyReLU()
        self.pff_layer.append(m)
        hid2_linear = paddle.nn.Linear(
            in_features=self.d_inner_hid,
            out_features=self.d_model,
            weight_attr=paddle.ParamAttr(
                #regularizer=L2Decay(coeff=0.0001),
                initializer=paddle.nn.initializer.Normal(
                    std=0.1 / math.sqrt(self.d_model))))
        self.add_sublayer('hid2_l', hid2_linear)
        self.pff_layer.append(hid2_linear)

        self.compute_qkv_layer = []
        q_linear = paddle.nn.Linear(
            in_features=self.d_model,
            out_features=d_key * n_head,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=0.1 /
                                                         math.sqrt(d_model))))
        self.add_sublayer("q_liner", q_linear)
        self.compute_qkv_layer.append(q_linear)

        k_linear = paddle.nn.Linear(
            in_features=self.d_model,
            out_features=d_key * n_head,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=0.1 /
                                                         math.sqrt(d_key))))
        self.add_sublayer("k_liner", k_linear)
        self.compute_qkv_layer.append(k_linear)

        v_linear = paddle.nn.Linear(
            in_features=self.d_model,
            out_features=d_value * n_head,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=0.1 /
                                                         math.sqrt(d_value))))
        self.add_sublayer("v_liner", v_linear)
        self.compute_qkv_layer.append(v_linear)

        po_linear = paddle.nn.Linear(
            in_features=d_model,
            out_features=d_model,
            weight_attr=paddle.ParamAttr(
                #regularizer=L2Decay(coeff=0.0001),
                initializer=paddle.nn.initializer.Normal(std=0.1 /
                                                         math.sqrt(d_model))))
        self.add_sublayer("po_liner", po_linear)
        self.compute_qkv_layer.append(po_linear)

    def positionwise_feed_forward(self, x, dropout_rate):
        """
        Position-wise Feed-Forward Networks.
        This module consists of two linear transformations with a ReLU activation
        in between, which is applied to each position separately and identically.
        """
        pff_input = x
        for _layer in self.pff_layer:
            pff_input = _layer(pff_input)

        if dropout_rate:
            pff_input = self.drop_out(pff_input)

        return pff_input

    def pre_post_process_layer_(self,
                                prev_out,
                                out,
                                process_cmd,
                                dropout_rate=0.5):
        """
        Add residual connection, layer normalization and droput to the out tensor
        optionally according to the value of process_cmd.
        This will be used before or after multi-head attention and position-wise
        feed-forward networks.
        """
        out = paddle.add(out, prev_out)
        for cmd in process_cmd:
            if cmd == "n":  # add layer normalization
                out = paddle.static.nn.layer_norm(
                    out,
                    begin_norm_axis=len(out.shape) - 1,
                    param_attr=paddle.nn.initializer.Constant(value=1.0),
                    bias_attr=paddle.nn.initializer.Constant(value=0.0))

            elif cmd == "d":  # add dropout
                if dropout_rate:
                    out = self.drop_out(out)
        return out

    def pre_post_process_layer(self, out, process_cmd, dropout_rate=0.5):
        """
        Add residual connection, layer normalization and droput to the out tensor
        optionally according to the value of process_cmd.
        This will be used before or after multi-head attention and position-wise
        feed-forward networks.
        """
        for cmd in process_cmd:
            if cmd == "a":  # add residual connection
                out = out
            elif cmd == "n":  # add layer normalization
                out = paddle.static.nn.layer_norm(
                    out,
                    begin_norm_axis=len(out.shape) - 1,
                    param_attr=paddle.nn.initializer.Constant(value=1.0),
                    bias_attr=paddle.nn.initializer.Constant(value=0.0))

            elif cmd == "d":  # add dropout
                if dropout_rate:
                    out = self.drop_out(out)
        return out

    def multi_head_attention(self, queries, keys, values, d_key, d_value,
                             d_model, n_head, dropout_rate):
        keys = queries if keys is None else keys
        values = keys if values is None else values
        #print(keys.shape)
        if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3
                ):
            raise ValueError(
                "Inputs: quries, keys and values should all be 3-D tensors.")

        def __compute_qkv(queries, keys, values):
            """
            Add linear projection to queries, keys, and values.
            """

            q = self.compute_qkv_layer[0](queries)
            k = self.compute_qkv_layer[1](keys)
            v = self.compute_qkv_layer[2](values)
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
                weights = self.drop_out(weights)
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

        q, k, v = __compute_qkv(queries, keys, values)
        q, k, v = __split_heads_qkv(q, k, v, n_head, d_key, d_value)

        ctx_multiheads = scaled_dot_product_attention(q, k, v, d_model,
                                                      dropout_rate)

        out = __combine_heads(ctx_multiheads)

        proj_out = self.compute_qkv_layer[3](out)

        return proj_out

    def encoder_layer(self, x):

        attention_out = self.multi_head_attention(
            self.pre_post_process_layer(x, self.preprocess_cmd,
                                        self.prepostprocess_dropout), None,
            None, self.d_key, self.d_value, self.d_model, self.n_head,
            self.dropout_rate)
        attn_output = self.pre_post_process_layer_(x, attention_out,
                                                   self.postprocess_cmd,
                                                   self.prepostprocess_dropout)

        ffd_output = self.positionwise_feed_forward(attn_output,
                                                    self.dropout_rate)

        return self.pre_post_process_layer_(attn_output, ffd_output,
                                            self.preprocess_cmd,
                                            self.prepostprocess_dropout)

    def forward(self, userid, hist_item_seq, hist_cat_seq, position_seq,
                target_item, target_cat, target_position):

        user_emb = self.userid_attr(userid)

        hist_item_emb = self.hist_item_emb_attr(hist_item_seq)

        hist_cat_emb = self.hist_cat_emb_attr(hist_cat_seq)

        hist_position_emb = self.hist_position_emb_attr(position_seq)

        target_item_emb = self.target_item_emb_attr(target_item)

        target_cat_emb = self.target_cat_emb_attr(target_cat)

        target_position_emb = self.target_position_emb_attr(target_position)

        item_sequence = paddle.concat(
            [hist_item_emb, hist_cat_emb, hist_position_emb], axis=2)
        target_sequence = paddle.concat(
            [target_item_emb, target_cat_emb, target_position_emb], axis=2)

        #print(position_sequence_target.shape)
        whole_embedding = paddle.concat(
            [item_sequence, target_sequence], axis=1)
        #print(whole_embedding)
        enc_output = whole_embedding
        '''
        for _ in range(self.n_encoder_layers):
            enc_output = self.encoder_layer(enc_output)
        '''
        enc_output = self.encoder_layer(enc_output)
        enc_output = self.pre_post_process_layer(
            enc_output, self.preprocess_cmd, self.prepostprocess_dropout)
        _concat = paddle.concat([user_emb, enc_output], axis=1)
        dnn_input = _concat
        for n_layer in self._dnn_layers:
            dnn_input = n_layer(dnn_input)
        dnn_input = paddle.sum(x=dnn_input, axis=1)
        return dnn_input
