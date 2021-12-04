#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""BERT4Rec model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import json
import math
import paddle
import paddle.nn as nn


class BertModel(nn.Layer):
    def __init__(self, _emb_size, _n_layer, _n_head, _voc_size,
                 _max_position_seq_len, _sent_types, hidden_act, _dropout,
                 _attention_dropout, initializer_range):
        super(BertModel, self).__init__()
        self._emb_size = _emb_size
        self._n_layer = _n_layer
        self._n_head = _n_head
        self._voc_size = _voc_size
        self._max_position_seq_len = _max_position_seq_len
        self._sent_types = _sent_types
        hidden_act = hidden_act
        if hidden_act == "gelu":
            self._hidden_act = nn.GELU()
        else:
            self._hidden_act = nn.ReLU()
        self._dropout = _dropout
        self._attention_dropout = _attention_dropout

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._dtype = "float32"

        self._param_initializer = nn.initializer.TruncatedNormal(
            std=initializer_range)

        self.word_emb = nn.Embedding(
            num_embeddings=self._voc_size,
            embedding_dim=self._emb_size,
            name=self._word_emb_name,
            weight_attr=paddle.ParamAttr(initializer=self._param_initializer),
            sparse=False)
        self.position_emb = nn.Embedding(
            num_embeddings=self._max_position_seq_len,
            embedding_dim=self._emb_size,
            weight_attr=paddle.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer),
            sparse=False)
        self.sent_emb = nn.Embedding(
            num_embeddings=self._sent_types,
            embedding_dim=self._emb_size,
            weight_attr=paddle.ParamAttr(
                name=self._sent_emb_name, initializer=self._param_initializer),
            sparse=False)
        self.enc_pre_process_layer = NormalizeDropLayer(
            self._dropout, self._emb_size, name='pre_encoder')
        self._enc_out_layer = Encoder(
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            attention_dropout=self._attention_dropout,
            hidden_act=self._hidden_act,
            param_initializer=self._param_initializer,
            name='encoder')
        self.mask_trans_feat = nn.Linear(
            in_features=self._emb_size,
            out_features=self._emb_size,
            weight_attr=paddle.ParamAttr(
                name="mask_lm_trans_fc.w_0",
                initializer=self._param_initializer),
            bias_attr=paddle.ParamAttr(name='mask_lm_trans_fc.b_0'))
        self.mask_trans_act = self._hidden_act
        self.mask_post_process_layer = NormalizeLayer(
            self._emb_size, name='mask_lm_trans')
        self.mask_lm_out_bias = self.create_parameter(
            shape=[self._voc_size],
            dtype=self._dtype,
            attr=paddle.ParamAttr(
                name="mask_lm_out_fc.b_0",
                initializer=paddle.nn.initializer.Constant(value=0.0)),
            is_bias=True)

    def forward(self, src_ids, position_ids, sent_ids, input_mask, mask_pos):
        emb_out = self.word_emb(src_ids)
        position_embs_out = self.position_emb(position_ids)
        emb_out = emb_out + position_embs_out
        sent_emb_out = self.sent_emb(sent_ids)
        emb_out = emb_out + sent_emb_out
        emb_out = self.enc_pre_process_layer(emb_out)
        if self._dtype == "float16":
            input_mask = paddle.cast(x=input_mask, dtype=self._dtype)
        else:
            input_mask = paddle.cast(x=input_mask, dtype='float32')
        self_attn_mask = paddle.matmul(
            x=input_mask, y=input_mask, transpose_y=True)

        self_attn_mask = paddle.scale(
            x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = paddle.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True
        self._enc_out = self._enc_out_layer(
            enc_input=emb_out, attn_bias=n_head_self_attn_mask)
        mask_pos = paddle.cast(x=mask_pos, dtype='int32')
        reshaped_emb_out = paddle.reshape(
            x=self._enc_out, shape=[-1, self._emb_size])
        mask_feat = paddle.gather(x=reshaped_emb_out, index=mask_pos, axis=0)
        mask_trans_feat_out = self.mask_trans_feat(mask_feat)
        mask_trans_feat_out = self.mask_trans_act(mask_trans_feat_out)
        mask_trans_feat_out = self.mask_post_process_layer(
            out=mask_trans_feat_out)
        for name, param in self.named_parameters():
            if name == "word_emb.weight":
                y_tensor = param
                break
        fc_out = paddle.matmul(
            x=mask_trans_feat_out, y=y_tensor, transpose_y=True)
        fc_out += self.mask_lm_out_bias
        return fc_out


class MultiHeadAttention(nn.Layer):
    def __init__(self,
                 d_key,
                 d_value,
                 d_model,
                 n_head=1,
                 dropout_rate=0.,
                 param_initializer=None,
                 name='multi_head_att'):
        super(MultiHeadAttention, self).__init__()
        self.q_linear = nn.Linear(
            in_features=d_model,
            out_features=d_key * n_head,
            weight_attr=paddle.ParamAttr(
                name=name + '_query_fc.w_0', initializer=param_initializer),
            bias_attr=name + '_query_fc.b_0')
        self.k_linear = nn.Linear(
            in_features=d_model,
            out_features=d_key * n_head,
            weight_attr=paddle.ParamAttr(
                name=name + '_key_fc.w_0', initializer=param_initializer),
            bias_attr=name + '_key_fc.b_0')
        self.v_linear = nn.Linear(
            in_features=d_model,
            out_features=d_value * n_head,
            weight_attr=paddle.ParamAttr(
                name=name + '_value_fc.w_0', initializer=param_initializer),
            bias_attr=name + '_value_fc.b_0')

        self.out_linear = nn.Linear(
            in_features=d_key * n_head,
            out_features=d_model,
            weight_attr=paddle.ParamAttr(
                name=name + '_output_fc.w_0', initializer=param_initializer),
            bias_attr=name + '_output_fc.b_0')
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate

    def forward(self, queries, keys, values, attn_bias):
        keys = queries if keys is None else keys
        values = keys if values is None else values

        q = self.q_linear(queries)
        k = self.k_linear(keys)
        v = self.v_linear(values)

        hidden_size = q.shape[-1]

        q = paddle.reshape(
            x=q, shape=[0, 0, self.n_head, hidden_size // self.n_head])
        q = paddle.transpose(
            x=q, perm=[0, 2, 1, 3]
        )  # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        k = paddle.reshape(
            x=k, shape=[0, 0, self.n_head, hidden_size // self.n_head])
        k = paddle.transpose(
            x=k, perm=[0, 2, 1, 3]
        )  # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        v = paddle.reshape(
            x=v, shape=[0, 0, self.n_head, hidden_size // self.n_head])
        v = paddle.transpose(
            x=v, perm=[0, 2, 1, 3]
        )  # [batch_size, n_head, max_sequence_len, hidden_size_per_head]

        # scale dot product attention
        attention_scores = paddle.matmul(x=q, y=k, transpose_y=True)
        product = paddle.multiply(
            attention_scores,
            paddle.to_tensor(
                1.0 / math.sqrt(float(self.d_key)), dtype='float32'))

        if attn_bias is not None:
            product += attn_bias
        weights = nn.functional.softmax(product)
        if self.dropout_rate:
            weights = nn.functional.dropout(
                weights,
                p=self.dropout_rate,
                mode="upscale_in_train",
                training=self.training)
        out = paddle.matmul(weights, v)
        out = paddle.transpose(out, perm=[0, 2, 1, 3])
        out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
        out = self.out_linear(out)
        return out


class NormalizeLayer(nn.Layer):
    def __init__(self, norm_shape=768, name=''):
        super(NormalizeLayer, self).__init__()
        self.name = name
        self.LayerNormal = nn.LayerNorm(
            norm_shape,
            epsilon=1e-05,
            weight_attr=paddle.ParamAttr(
                name=self.name + '_layer_norm_scale',
                initializer=nn.initializer.Constant(1.)),
            bias_attr=paddle.ParamAttr(
                name=self.name + '_layer_norm_bias',
                initializer=nn.initializer.Constant(0.)))

    def forward(self, out):
        out_dtype = out.dtype
        if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
            out = paddle.cast(x=out, dtype="float32")
        out = self.LayerNormal(out)
        if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
            out = paddle.cast(x=out, dtype="float16")
        return out


class NormalizeDropLayer(nn.Layer):
    def __init__(self, dropout_rate=0., norm_shape=768, name=''):
        super(NormalizeDropLayer, self).__init__()
        self.name = name
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate, mode="upscale_in_train")
        self.LayerNormal = nn.LayerNorm(
            norm_shape,
            epsilon=1e-05,
            weight_attr=paddle.ParamAttr(
                name=self.name + '_layer_norm_scale',
                initializer=nn.initializer.Constant(1.)),
            bias_attr=paddle.ParamAttr(
                name=self.name + '_layer_norm_bias',
                initializer=nn.initializer.Constant(0.)))

    def forward(self, out):
        out_dtype = out.dtype
        if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
            out = paddle.cast(x=out, dtype="float32")
        out = self.LayerNormal(out)
        if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
            out = paddle.cast(x=out, dtype="float16")
        if self.dropout_rate:
            out = self.dropout(out)
        return out


class DropResidualNormalizeLayer(nn.Layer):
    def __init__(self, dropout_rate=0., norm_shape=768, name=''):
        super(DropResidualNormalizeLayer, self).__init__()
        self.name = name
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate, mode="upscale_in_train")
        self.LayerNormal = nn.LayerNorm(
            norm_shape,
            epsilon=1e-05,
            weight_attr=paddle.ParamAttr(
                name=self.name + '_layer_norm_scale',
                initializer=nn.initializer.Constant(1.)),
            bias_attr=paddle.ParamAttr(
                name=self.name + '_layer_norm_bias',
                initializer=nn.initializer.Constant(0.)))

    def forward(self, out, prev_out=None):
        if self.dropout_rate:
            out = self.dropout(out)
        if prev_out is not None:
            out = out + prev_out
        out_dtype = out.dtype
        if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
            out = paddle.cast(x=out, dtype="float32")
        out = self.LayerNormal(out)
        if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
            out = paddle.cast(x=out, dtype="float16")
        return out


class FFN(nn.Layer):
    def __init__(self,
                 d_inner_hid,
                 d_hid,
                 hidden_act,
                 param_initializer=None,
                 name='ffn'):
        super(FFN, self).__init__()

        self.fc1 = nn.Linear(
            in_features=d_hid,
            out_features=d_inner_hid,
            weight_attr=paddle.ParamAttr(
                name=name + '_fc_0.w_0', initializer=param_initializer),
            bias_attr=name + '_fc_0.b_0')
        self.hidden_act = hidden_act
        self.fc2 = nn.Linear(
            in_features=d_inner_hid,
            out_features=d_hid,
            weight_attr=paddle.ParamAttr(
                name=name + '_fc_1.w_0', initializer=param_initializer),
            bias_attr=name + '_fc_1.b_0')

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.hidden_act(hidden)
        out = self.fc2(hidden)
        return out


class EncoderLayer(nn.Layer):
    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 attention_dropout,
                 hidden_act,
                 param_initializer=None,
                 name=''):
        super(EncoderLayer, self).__init__()
        self.multi_head_attn = MultiHeadAttention(
            d_key,
            d_value,
            d_model,
            n_head,
            attention_dropout,
            param_initializer=param_initializer,
            name=name + '_multi_head_att')

        self.drop_residual_normalize_layer_1 = DropResidualNormalizeLayer(
            attention_dropout, norm_shape=d_model, name=name + '_post_att')

        self.positionwise_feed_layer = FFN(d_inner_hid,
                                           d_model,
                                           hidden_act,
                                           param_initializer,
                                           name=name + '_ffn')
        self.drop_residual_normalize_layer_2 = DropResidualNormalizeLayer(
            attention_dropout, norm_shape=d_model, name=name + '_post_ffn')

    def forward(self, enc_input, attn_bias):
        multi_output = self.multi_head_attn(
            queries=enc_input, keys=None, values=None, attn_bias=attn_bias)
        attn_output = self.drop_residual_normalize_layer_1(
            prev_out=enc_input, out=multi_output)
        ffd_output = self.positionwise_feed_layer(attn_output)
        out = self.drop_residual_normalize_layer_2(
            prev_out=attn_output, out=ffd_output)

        return out


class Encoder(nn.Layer):
    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 attention_dropout,
                 hidden_act,
                 param_initializer=None,
                 name=''):
        super(Encoder, self).__init__()
        self.encoder_layer = nn.LayerList([
            EncoderLayer(n_head, d_key, d_value, d_model, d_inner_hid,
                         attention_dropout, hidden_act, param_initializer,
                         name + '_layer_' + str(i)) for i in range(n_layer)
        ])

    def forward(self, enc_input, attn_bias):
        enc_output = None
        for enc in self.encoder_layer:
            enc_output = enc(enc_input, attn_bias)
            enc_input = enc_output
        return enc_output


class BertConfig(object):
    """ 根据config_path来读取网络的配置 """

    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing bert model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')
