# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import paddle
import paddle.nn as nn
import math
import numpy as np

class AttentionPooling(nn.Layer):
    def __init__(self, hidden_size, initializer_range):
        super(AttentionPooling, self).__init__()
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.att_fc1 = paddle.nn.Linear(self.hidden_size, self.hidden_size)
        self.att_fc2 = paddle.nn.Linear(self.hidden_size, 1)
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range,
                        shape=layer.weight.shape))
            layer.bias.set_value(paddle.full(shape=layer.bias.shape, fill_value=0.0))

    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.functional.tanh(e)
        alpha = self.att_fc2(e)
        alpha = paddle.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (paddle.sum(alpha, axis=1, keepdim=True) + 1e-8)
        x = paddle.bmm(x.transpose([0, 2, 1]), alpha)
        x = paddle.reshape(x, (bz, -1))
        return x


class FastSelfAttention(nn.Layer):
    def __init__(self, hidden_size, num_attention_heads, initializer_range):
        super(FastSelfAttention, self).__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (hidden_size, num_attention_heads))
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.num_attention_heads = num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim = hidden_size
        self.initializer_range = initializer_range

        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(axis=-1)

        self.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range,
                        shape=layer.weight.shape))
            layer.bias.set_value(paddle.full(shape=layer.bias.shape, fill_value=0.0))

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [self.num_attention_heads] + [self.attention_head_size]
        x = paddle.reshape(x, new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(self, hidden_states, attention_mask):
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)

        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose([0, 2, 1]) / self.attention_head_size**0.5

        # add attention mask
        query_for_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # breakpoint()
        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = paddle.matmul(query_weight, query_layer).transpose([0, 2, 1, 3]).reshape([-1, 1, self.num_attention_heads * self.attention_head_size])
        pooled_query_repeat = pooled_query.tile([1, seq_len, 1])
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer = mixed_key_layer * pooled_query_repeat

        query_key_score = (self.key_att(mixed_query_key_layer) / self.attention_head_size**0.5).transpose([0, 2, 1])
        # add attention mask
        query_key_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = paddle.matmul(query_key_weight, key_layer)

        # query = value
        weighted_value = (pooled_key * query_layer).transpose([0, 2, 1, 3])
        weighted_value = paddle.reshape(weighted_value, 
            weighted_value.shape[:-2] + [self.num_attention_heads * self.attention_head_size])
        weighted_value = self.transform(weighted_value) + mixed_query_layer

        return weighted_value


class FastAttention(paddle.nn.Layer):
    def __init__(self, hidden_size, num_attention_heads, initializer_range, 
                 layer_norm_eps, hidden_dropout_prob):
        super(FastAttention, self).__init__()
        self.self_attention = FastSelfAttention(hidden_size, num_attention_heads, initializer_range)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self_attention(input_tensor, attention_mask)
        self_output = self.dense(self_output)
        self_output = self.dropout(self_output)
        attention_output = self.layer_norm(self_output + input_tensor)

        return attention_output


class FastformerLayer(paddle.nn.Layer):
    def __init__(self, hidden_size, num_attention_heads, initializer_range, layer_norm_eps, intermediate_size, hidden_dropout_prob):
        super(FastformerLayer, self).__init__()
        self.attention = FastAttention(hidden_size, 
                                       num_attention_heads, 
                                       initializer_range, 
                                       layer_norm_eps, 
                                       hidden_dropout_prob
                                      )

        # BERT Intermediate
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

        # BERT Output
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.dense1(attention_output)
        intermediate_output = self.intermediate_act_fn(hidden_states)
        intermediate_output = self.dense2(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)
        layer_output = self.layer_norm(intermediate_output + attention_output)

        return layer_output


class FastformerEncoder(paddle.nn.Layer):
    def __init__(self, hidden_size, num_attention_heads, initializer_range, 
                 layer_norm_eps, intermediate_size, hidden_dropout_prob, 
                 num_hidden_layers, max_position_embeddings, pooler_type, pooler_count):
        super(FastformerEncoder, self).__init__()
        self.initializer_range = initializer_range
        self.encoders = nn.LayerList([FastformerLayer(hidden_size, 
                                                      num_attention_heads, 
                                                      initializer_range, 
                                                      layer_norm_eps, 
                                                      intermediate_size, 
                                                      hidden_dropout_prob) for _ in range(num_hidden_layers)])
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # support multiple different poolers with shared bert encoder.
        self.poolers = nn.LayerList()
        if pooler_type == 'weightpooler':
            for _ in range(pooler_count):
                self.poolers.append(AttentionPooling(hidden_size, initializer_range))
        logging.info(f"This model has {len(self.poolers)} poolers.")

        self.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            layer.weight.set_value(paddle.tensor.normal(
                                    mean=0.0,
                                    std=self.initializer_range,
                                    shape=layer.weight.shape))
            if isinstance(layer, (nn.Embedding)) and layer._padding_idx is not None:
                with paddle.no_grad():
                    layer.weight[layer._padding_idx].fill_(0)
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.full(shape=layer.bias.shape, fill_value=0.0))
            layer.weight.set_value(paddle.full(shape=layer.weight.shape, fill_value=0.0))
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            layer.bias.set_value(paddle.full(shape=layer.bias.shape, fill_value=0.0))

    def forward(self, input_embs, attention_mask, pooler_index=0):
        # input_embs: batch_size, seq_len, emb_dim
        # attention_mask: batch_size, seq_len, emb_dim

        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.astype(next(iter(self.parameters())).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embeddings = input_embs + position_embeddings
        embeddings = input_embs
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        # print(embeddings.size())
        all_hidden_states = [embeddings]

        for i, layer_module in enumerate(self.encoders):
            layer_outputs = layer_module(all_hidden_states[-1], extended_attention_mask)
            all_hidden_states.append(layer_outputs)
        assert len(self.poolers) > pooler_index
        output = self.poolers[pooler_index](all_hidden_states[-1], attention_mask)

        return output


class Fastformer(paddle.nn.Layer):

    def __init__(self, hidden_size, num_attention_heads, initializer_range, 
                 layer_norm_eps, intermediate_size, hidden_dropout_prob, 
                 num_hidden_layers, max_position_embeddings, pooler_type, vocab_size, pooler_count=1):
        super(Fastformer, self).__init__()
        self.initializer_range = initializer_range
        # self.word_embedding = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.fastformer_model = FastformerEncoder(hidden_size, num_attention_heads, initializer_range, 
                 layer_norm_eps, intermediate_size, hidden_dropout_prob, 
                 num_hidden_layers, max_position_embeddings, pooler_type, pooler_count)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            layer.weight.set_value(paddle.tensor.normal(
                                    mean=0.0,
                                    std=self.initializer_range,
                                    shape=layer.weight.shape))
            if isinstance(layer, (nn.Embedding)) and layer._padding_idx is not None:
                with paddle.no_grad():
                    layer.weight[layer._padding_idx].fill_(0)
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            layer.bias.set_value(paddle.full(shape=layer.bias.shape, fill_value=0.0))

    def forward(self, inputs, mask):
        text_vec = self.fastformer_model(inputs, mask)
        return text_vec


if __name__ == '__main__':

    batch_size = 48
    seq_len = 30
    hidden_size = 256
    hidden_dropout_prob = 0.2
    num_hidden_layers =  2
    hidden_act = "gelu"
    num_attention_heads = 16
    intermediate_size = 256
    max_position_embeddings = 256
    type_vocab_size = 2
    vocab_size = 100000
    layer_norm_eps = 1e-12
    initializer_range = 0.02
    pooler_type = "weightpooler"
    enable_fp16 =  False

    # Test FastAttentionPooling
    # x = paddle.rand(shape=[4, seq_len, hidden_size])
    # model = AttentionPooling(hidden_size, initializer_range)
    # out = model(x)
    # print(out.shape)
    # breakpoint()

    # Test FastAttentionPooling

    # batch_size, log_length, news_dim
    x = paddle.paddle.randint(low=0, high=100, shape=[batch_size, hidden_size])
    x = paddle.tril(x)

    word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
    embedding = word_embedding(x)

    attention_mask = paddle.cast(x, dtype='bool')
    attention_mask = paddle.cast(attention_mask, dtype='float32')

    model = Fastformer(hidden_size, num_attention_heads, initializer_range, 
                       layer_norm_eps, intermediate_size, hidden_dropout_prob, 
                       num_hidden_layers, max_position_embeddings, pooler_type, vocab_size
                      )
    out = model(embedding, attention_mask)
    print(out.shape)