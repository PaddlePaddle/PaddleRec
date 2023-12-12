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
from fastformer import Fastformer
import math
import numpy as np


class AttentionPooling(nn.Layer):
    def __init__(self, d_h, hidden_size, drop_rate):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size // 2)
        self.att_fc2 = nn.Linear(hidden_size // 2, 1)
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)  # (bz, seq_len, hidden_size // 2)
        e = paddle.tanh(e)
        alpha = self.att_fc2(e)  # (bz, seq_len, 1)

        alpha = paddle.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (paddle.sum(alpha, axis=1, keepdim=True) + 1e-8)

        x = paddle.bmm(x.transpose([0, 2, 1]), alpha)
        x = paddle.reshape(x, (bz, -1))  # (bz, hidden_size)
        return x


class NewsEncoder(nn.Layer):
    def __init__(
        self,
        article_content_size,
        article_title_size,
        browse_size,
        neg_condidate_sample_size,
        word_dimension,
        category_size,
        sub_category_size,
        cate_dimension,
        word_dict_size,
        hidden_size,
        num_attention_heads,
        initializer_range,
        layer_norm_eps,
        intermediate_size,
        hidden_dropout_prob,
        num_hidden_layers,
        max_position_embeddings,
        pooler_type,
    ):
        super(NewsEncoder, self).__init__()

        self.article_content_size = article_content_size
        self.article_title_size = article_title_size
        self.browse_size = browse_size
        self.neg_condidate_sample_size = neg_condidate_sample_size
        self.word_dimension = word_dimension
        self.category_size = category_size
        self.sub_category_size = sub_category_size
        self.cate_dimension = cate_dimension
        self.word_dict_size = word_dict_size
        self.conv_out_channel_size = 400
        self.attention_projection_size = 100

        self.load_word_embedding()
        self.attention_vec = []
        self.attention_layer = []

        self.cate_embedding = paddle.nn.Embedding(
            self.category_size + 1,
            self.cate_dimension,
            weight_attr=paddle.ParamAttr(
                name="cate_embedding", initializer=paddle.nn.initializer.Normal(std=0.1)
            ),
        )
        self.sub_cate_embedding = paddle.nn.Embedding(
            self.sub_category_size + 1,
            self.cate_dimension,
            weight_attr=paddle.ParamAttr(
                name="sub_cate_embedding",
                initializer=paddle.nn.initializer.Normal(std=0.1),
            ),
        )
        # title_emb [batch, word_emb_d, title_size]
        self.conv_title = nn.Conv1D(
            self.word_dimension, self.conv_out_channel_size, 3, padding="same"
        )
        self.conv_content = nn.Conv1D(
            self.word_dimension, self.conv_out_channel_size, 3, padding="same"
        )
        self.conv_title_bias = paddle.create_parameter(
            shape=[self.conv_out_channel_size],
            dtype="float32",
            name="conv_title_bias",
            default_initializer=paddle.nn.initializer.Normal(
                std=1.0 / self.conv_out_channel_size
            ),
        )
        self.add_parameter("conv_title_bias", self.conv_title_bias)
        self.conv_content_bias = paddle.create_parameter(
            shape=[self.conv_out_channel_size],
            dtype="float32",
            name="conv_content_bias",
            default_initializer=paddle.nn.initializer.Normal(
                std=1.0 / self.conv_out_channel_size
            ),
        )
        self.add_parameter("conv_content_bias", self.conv_content_bias)
        self.category_linear = paddle.nn.Linear(
            in_features=self.cate_dimension,
            out_features=self.conv_out_channel_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=0.1)
            ),
        )
        self.add_sublayer("category_linear", self.category_linear)
        self.sub_category_linear = paddle.nn.Linear(
            in_features=self.cate_dimension,
            out_features=self.conv_out_channel_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=0.1)
            ),
        )
        self.add_sublayer("sub_category_linear", self.sub_category_linear)

        # Fastformer Attention
        # self.mix_attention = Fastformer(
        #     hidden_size=self.conv_out_channel_size,
        #     num_attention_heads=num_attention_heads,
        #     initializer_range=initializer_range,
        #     layer_norm_eps=layer_norm_eps,
        #     intermediate_size=intermediate_size,
        #     hidden_dropout_prob=hidden_dropout_prob,
        #     num_hidden_layers=num_hidden_layers,
        #     max_position_embeddings=max_position_embeddings,
        #     pooler_type=pooler_type,
        # )

        self.mix_attention = self.make_attention_layer(
            "mix_attention",
            [self.conv_out_channel_size, self.attention_projection_size],
        )

        self.title_attention = Fastformer(
            hidden_size=self.conv_out_channel_size,
            num_attention_heads=num_attention_heads,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            pooler_type=pooler_type,
        )

        self.content_attention = Fastformer(
            hidden_size=self.conv_out_channel_size,
            num_attention_heads=num_attention_heads,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            pooler_type=pooler_type,
        )

    def load_word_embedding(self):
        self.word2vec_embedding = paddle.nn.Embedding(
            self.word_dict_size + 1,
            self.word_dimension,
            weight_attr=paddle.ParamAttr(
                name="word2vec_embedding", initializer=paddle.nn.initializer.Uniform()
            ),
        )

    def create_mask(self, input_tensor):
        attention_mask = paddle.ones_like(input_tensor, dtype="int32")
        condition = paddle.equal(input_tensor, self.word_dict_size)
        attention_mask = paddle.logical_not(condition)
        attention_mask = paddle.cast(attention_mask, dtype="float32")
        return attention_mask

    def make_attention_layer(self, name_base, size):
        row = size[0]
        col = size[1]
        vec = paddle.create_parameter(
            shape=(col, 1),
            dtype="float32",
            name=name_base + "_vec_generated",
            default_initializer=paddle.nn.initializer.Normal(std=0.1),
        )
        self.add_parameter(name_base + "_vec_generated", vec)
        index = len(self.attention_vec)
        self.attention_vec.append(vec)
        linear = paddle.nn.Linear(
            in_features=row,
            out_features=col,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=0.01)
            ),
        )
        self.attention_layer.append(linear)
        self.add_sublayer(name_base + "_linear_generated", linear)

        def func(input):
            # input [b,g, row]
            # [b,g,col]
            project = self.attention_layer[index](input)
            # [b,g,1]
            project = paddle.matmul(project, self.attention_vec[index])
            # [b,1,g]
            project = paddle.transpose(project, perm=[0, 2, 1])
            weight = paddle.nn.functional.softmax(project)
            # [b, 1, row]
            output = paddle.matmul(weight, input)
            # [b,row]
            output = paddle.reshape(output, [-1, row])
            return output

        return func

    def forward(self, category, sub_category, title, content):
        cate_emb = self.cate_embedding(category)
        sub_cate_emb = self.sub_cate_embedding(sub_category)
        # [b, conv_out]
        category = paddle.nn.ReLU()(self.category_linear(cate_emb))
        sub_category = paddle.nn.ReLU()(self.sub_category_linear(sub_cate_emb))
        # title [batch, title_size]
        # title_emb [batch,title_size, word_emb_d]
        title_emb = self.word2vec_embedding(title)
        # title_emb [batch, word_emb_d, title_size]
        title_emb = paddle.transpose(title_emb, perm=[0, 2, 1])
        # title_emb [batch,conv_out,title_size]
        title_emb = self.conv_title(title_emb)
        # content_emb [batch, content_size, word_emb_d]
        content_emb = self.word2vec_embedding(content)
        # content_emb [batch, word_emb_d,content_size,]
        content_emb = paddle.transpose(content_emb, perm=[0, 2, 1])
        # [batch,conv_out,content_size]
        content_emb = self.conv_title(content_emb)
        # title_emb [batch,title_size,conv_out]
        # content_emb [batch, content_size, conv_out]
        title_emb = paddle.transpose(title_emb, perm=[0, 2, 1])
        content_emb = paddle.transpose(content_emb, perm=[0, 2, 1])
        title_emb = paddle.nn.ReLU()(paddle.add(title_emb, self.conv_title_bias))
        content_emb = paddle.nn.ReLU()(paddle.add(content_emb, self.conv_content_bias))

        # Create mask for titles and contents
        title_mask = self.create_mask(title)
        content_mask = self.create_mask(content)
        # breakpoint()

        # [b,conv_out]
        title_emb = self.title_attention(title_emb, title_mask)
        content_emb = self.content_attention(content_emb, content_mask)

        # [b,conv_out * 4]
        vec = paddle.concat([title_emb, content_emb, category, sub_category], axis=-1)
        # [b, 4, conv_out]
        vec_group = paddle.reshape(vec, [-1, 4, self.conv_out_channel_size])
        vec_group_mask = paddle.ones(
            shape=[vec_group.shape[0], vec_group.shape[1]], dtype="int32"
        )

        # breakpoint()
        # [b, conv_out]
        final_vec = self.mix_attention(vec_group)  # , vec_group_mask)
        return final_vec


class UserEncoder(nn.Layer):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        initializer_range,
        layer_norm_eps,
        intermediate_size,
        hidden_dropout_prob,
        num_hidden_layers,
        max_position_embeddings,
        pooler_type,
    ):
        super(UserEncoder, self).__init__()
        self.encoder = Fastformer(
            hidden_size,
            num_attention_heads,
            initializer_range,
            layer_norm_eps,
            intermediate_size,
            hidden_dropout_prob,
            num_hidden_layers,
            max_position_embeddings,
            pooler_type,
        )

    def forward(self, inputs, mask):
        user_vec = self.encoder(inputs, mask)
        return user_vec


class FastRecommender(paddle.nn.Layer):
    def __init__(
        self,
        article_content_size,
        article_title_size,
        browse_size,
        neg_condidate_sample_size,
        word_dimension,
        category_size,
        sub_category_size,
        cate_dimension,
        word_dict_size,
        hidden_size,
        num_attention_heads,
        initializer_range,
        layer_norm_eps,
        intermediate_size,
        hidden_dropout_prob,
        num_hidden_layers,
        max_position_embeddings,
        pooler_type,
    ):
        super(FastRecommender, self).__init__()

        self.article_content_size = article_content_size
        self.article_title_size = article_title_size
        self.browse_size = browse_size
        self.neg_condidate_sample_size = neg_condidate_sample_size
        self.word_dimension = word_dimension
        self.category_size = category_size
        self.sub_category_size = sub_category_size
        self.cate_dimension = cate_dimension
        self.word_dict_size = word_dict_size
        self.conv_out_channel_size = 400
        self.attention_projection_size = 100

        self.news_encoder = NewsEncoder(
            article_content_size,
            article_title_size,
            browse_size,
            neg_condidate_sample_size,
            word_dimension,
            category_size,
            sub_category_size,
            cate_dimension,
            word_dict_size,
            hidden_size,
            num_attention_heads,
            initializer_range,
            layer_norm_eps,
            intermediate_size,
            hidden_dropout_prob,
            num_hidden_layers,
            max_position_embeddings,
            pooler_type,
        )

        self.user_encoder = UserEncoder(
            hidden_size=self.conv_out_channel_size,
            num_attention_heads=num_attention_heads,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            pooler_type=pooler_type,
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, sparse_inputs):
        (
            cate_sample,
            cate_visit,
            sub_cate_sample,
            sub_cate_visit,
            title_sample,
            title_visit,
            content_sample,
            content_visit,
        ) = sparse_inputs[:]

        cate = paddle.concat([cate_sample, cate_visit], axis=-1)
        sub_cate = paddle.concat([sub_cate_sample, sub_cate_visit], axis=-1)
        title = paddle.concat([title_sample, title_visit], axis=-2)
        content = paddle.concat([content_sample, content_visit], axis=-2)
        # [b * (sample + visit)]
        cate = paddle.reshape(cate, [-1])
        # [b * (sample + visit)]
        sub_cate = paddle.reshape(sub_cate, [-1])
        # [b * (sample + visit), article_title_size]
        title = paddle.reshape(title, [-1, self.article_title_size])
        # [b * (sample + visit), article_content_size]
        content = paddle.reshape(content, [-1, self.article_content_size])

        # [b, (sample + visit) * conv_out_size]
        final_vec = self.news_encoder(cate, sub_cate, title, content)
        # [b, (sample + visit) , conv_out_size]
        final_vec = paddle.reshape(
            final_vec,
            [
                -1,
                self.neg_condidate_sample_size + 1 + self.browse_size,
                self.conv_out_channel_size,
            ],
        )
        # [b, sample, conv_out_size] [b,visit, conv_out_size]
        sample_emb, visit_emb = paddle.split(
            final_vec,
            num_or_sections=[self.neg_condidate_sample_size + 1, self.browse_size],
            axis=1,
        )

        visit_mask = paddle.ones(
            shape=[visit_emb.shape[0], visit_emb.shape[1]], dtype="int32"
        )

        # breakpoint()

        # [b, conv_out_size]
        visit_compressed_emb = self.user_encoder(visit_emb, visit_mask)
        # breakpoint()

        # [b, conv_out_size,1]
        visit_compressed_emb = paddle.reshape(
            visit_compressed_emb, [-1, self.conv_out_channel_size, 1]
        )

        # [b,sample,1]
        predict = paddle.matmul(sample_emb, visit_compressed_emb)

        # [b,sample]
        predict = paddle.reshape(predict, [-1, self.neg_condidate_sample_size + 1])
        # predict = paddle.nn.functional.softmax(predict)
        # predict = paddle.nn.functional.sigmoid(predict)
        # print(predict)
        # breakpoint()
        return predict


if __name__ == "__main__":
    batch_size = 48
    seq_len = 30
    hidden_size = 256
    hidden_dropout_prob = 0.2
    num_hidden_layers = 2
    num_attention_heads = 16
    intermediate_size = 256
    max_position_embeddings = 256
    type_vocab_size = 2
    vocab_size = 100000
    layer_norm_eps = 1e-12
    initializer_range = 0.02
    pooler_type = "weightpooler"
    enable_fp16 = False
    d_h = hidden_size
    # hidden_size = 256
    drop_rate = 0.2

    # from naml
    article_content_size = 30
    article_title_size = 10
    browse_size = 10
    neg_condidate_sample_size = 4
    word_dimension = 30
    category_size = 4
    sub_category_size = 10
    cate_dimension = 32
    word_dict_size = 101

    # Test AttentionPooling
    # batch_size, log_length, news_dim
    x = paddle.rand(shape=[batch_size, seq_len, hidden_size])
    x = paddle.tril(x)

    model = AttentionPooling(d_h, hidden_size, drop_rate)
    out = model(x)
    print(out.shape)

    # Test UserEncoder
    # batch_size, log_length, news_dim
    x = paddle.rand(shape=[batch_size, seq_len, hidden_size])
    x = paddle.tril(x)

    model = FastRecommender(
        article_content_size,
        article_title_size,
        browse_size,
        neg_condidate_sample_size,
        word_dimension,
        category_size,
        sub_category_size,
        cate_dimension,
        word_dict_size,
        hidden_size,
        num_attention_heads,
        initializer_range,
        layer_norm_eps,
        intermediate_size,
        hidden_dropout_prob,
        num_hidden_layers,
        max_position_embeddings,
        pooler_type,
    )

    # sparse_inputs =
    # out = model(sparse_inputs)
    # print(out.shape)
