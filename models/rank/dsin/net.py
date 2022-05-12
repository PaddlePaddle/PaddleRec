# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F
import math
import numpy as np
from sequence_layers import PositionalEncoder, AttentionSequencePoolingLayer, MLP


class DSIN_layer(nn.Layer):
    def __init__(
            self,
            user_size,
            adgroup_size,
            pid_size,
            cms_segid_size,
            cms_group_size,
            final_gender_size,
            age_level_size,
            pvalue_level_size,
            shopping_level_size,
            occupation_size,
            new_user_class_level_size,
            campaign_size,
            customer_size,
            cate_size,
            brand_size,  # above is all sparse feat size
            sparse_embed_size=4,
            att_embedding_size=8,
            sess_count=5,
            sess_max_length=10,
            l2_reg_embedding=1e-6):
        super().__init__()

        # feature size
        self.user_size = user_size
        self.adgroup_size = adgroup_size
        self.pid_size = pid_size
        self.cms_segid_size = cms_segid_size
        self.cms_group_size = cms_group_size
        self.final_gender_size = final_gender_size
        self.age_level_size = age_level_size
        self.pvalue_level_size = pvalue_level_size
        self.shopping_level_size = shopping_level_size
        self.occupation_size = occupation_size
        self.new_user_class_level_size = new_user_class_level_size
        self.campaign_size = campaign_size
        self.customer_size = customer_size
        self.cate_size = cate_size
        self.brand_size = brand_size

        # sparse embed size
        self.sparse_embed_size = sparse_embed_size

        # transform attention embed size
        self.att_embedding_size = att_embedding_size

        # hyper_parameters
        self.sess_count = 5
        self.sess_max_length = 10

        # sparse embedding layer
        self.userid_embeddings_var = paddle.nn.Embedding(
            self.user_size,
            self.sparse_embed_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.adgroup_embeddings_var = paddle.nn.Embedding(
            self.adgroup_size,
            self.sparse_embed_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.pid_embeddings_var = paddle.nn.Embedding(
            self.pid_size,
            self.sparse_embed_size,
            #sparse=True,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.cmsid_embeddings_var = paddle.nn.Embedding(
            self.cms_segid_size,
            self.sparse_embed_size,
            #sparse=True,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.cmsgroup_embeddings_var = paddle.nn.Embedding(
            self.cms_group_size,
            self.sparse_embed_size,
            #sparse=True,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.gender_embeddings_var = paddle.nn.Embedding(
            self.final_gender_size,
            self.sparse_embed_size,
            #sparse=True,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.age_embeddings_var = paddle.nn.Embedding(
            self.age_level_size,
            self.sparse_embed_size,
            #sparse=True,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.pvalue_embeddings_var = paddle.nn.Embedding(
            self.pvalue_level_size,
            self.sparse_embed_size,
            #sparse=True,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.shopping_embeddings_var = paddle.nn.Embedding(
            self.shopping_level_size,
            self.sparse_embed_size,
            #sparse=True,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.occupation_embeddings_var = paddle.nn.Embedding(
            self.occupation_size,
            self.sparse_embed_size,
            #sparse=True,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.new_user_class_level_embeddings_var = paddle.nn.Embedding(
            self.new_user_class_level_size,
            self.sparse_embed_size,
            #sparse=True,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.campaign_embeddings_var = paddle.nn.Embedding(
            self.campaign_size,
            self.sparse_embed_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.customer_embeddings_var = paddle.nn.Embedding(
            self.customer_size,
            self.sparse_embed_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.cate_embeddings_var = paddle.nn.Embedding(
            self.cate_size,
            self.sparse_embed_size,
            sparse=True,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        self.brand_embeddings_var = paddle.nn.Embedding(
            self.brand_size,
            self.sparse_embed_size,
            sparse=True,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(l2_reg_embedding),
                initializer=nn.initializer.Normal(
                    mean=0.0, std=0.0001)))

        # sess interest extractor layer
        self.position_encoding = PositionalEncoder(2 * self.sparse_embed_size)
        self.transform = nn.TransformerEncoderLayer(
            d_model=self.att_embedding_size,
            nhead=8,
            dim_feedforward=64,
            weight_attr=self._get_weight_attr(),
            bias_attr=False,
            dropout=0.0)

        # sess interest interacting layer
        self.bilstm = nn.LSTM(
            2 * self.sparse_embed_size,
            2 * self.sparse_embed_size,
            num_layers=2,
            direction='bidirectional')

        # sess interest activating layer
        self.transform_actpool = AttentionSequencePoolingLayer(
            weight_normalization=True, name='transform')
        self.lstm_actpool = AttentionSequencePoolingLayer(
            weight_normalization=True, name='lstm')

        # MLP moudle
        self.mlp = MLP(mlp_hidden_units=[77, 200, 80])

    def _get_weight_attr(self):
        return paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(
            std=0.05))

    def forward(self, inputs):
        '''
        inputs : tulpe, (sparse_input, dense_input, sess_input, sess_length)
            sparse_input: (N, 15)
            dense_input: (N,)
            sess_input：(N, 10, 10)
            sess_length: (N,)
        '''
        sparse_input, dense_input, sess_input, sess_length = inputs
        #assert(type(sess_length) == paddle.Tensor), f"At Attention SequencePoolingLayer expected inputs[2]'s type is paddle.Tensor, but got {type(sess_length)}"

        # sparse and dense feature
        self.user = sparse_input[:, 0]
        self.adgroup = sparse_input[:, 1]
        self.pid = sparse_input[:, 2]
        self.cmsid = sparse_input[:, 3]
        self.cmsgroup = sparse_input[:, 4]
        self.gender = sparse_input[:, 5]
        self.age = sparse_input[:, 6]
        self.pvalue = sparse_input[:, 7]
        self.shopping = sparse_input[:, 8]
        self.occupation = sparse_input[:, 9]
        self.new_user_class = sparse_input[:, 10]
        self.campaign = sparse_input[:, 11]
        self.customer = sparse_input[:, 12]
        self.cate = sparse_input[:, 13]
        self.brand = sparse_input[:, 14]
        self.price = dense_input.unsqueeze_(-1)

        # sparse feature embedding
        self.user_embeded = self.userid_embeddings_var(self.user)
        self.adgroup_embeded = self.adgroup_embeddings_var(self.adgroup)
        self.pid_embeded = self.pid_embeddings_var(self.pid)
        self.cmsid_embeded = self.cmsid_embeddings_var(self.cmsid)
        self.cmsgroup_embeded = self.cmsgroup_embeddings_var(self.cmsgroup)
        self.gender_embeded = self.gender_embeddings_var(self.gender)
        self.age_embeded = self.age_embeddings_var(self.age)
        self.pvalue_embeded = self.pvalue_embeddings_var(self.pvalue)
        self.shopping_embeded = self.shopping_embeddings_var(self.shopping)
        self.occupation_embeded = self.occupation_embeddings_var(
            self.occupation)
        self.new_user_class_embeded = self.new_user_class_level_embeddings_var(
            self.new_user_class)
        self.campaign_embeded = self.campaign_embeddings_var(self.campaign)
        self.customer_embeded = self.customer_embeddings_var(self.customer)
        self.cate_embeded = self.cate_embeddings_var(self.cate)
        self.brand_embeded = self.brand_embeddings_var(self.brand)

        # concat query embeded  
        # Note: query feature is cate_embeded and brand_embeded
        query_embeded = paddle.concat([self.cate_embeded, self.brand_embeded],
                                      -1)

        # concat sparse feature embeded  
        deep_input_embeded = paddle.concat([
            self.user_embeded, self.adgroup_embeded, self.pid_embeded,
            self.cmsid_embeded, self.cmsgroup_embeded, self.gender_embeded,
            self.age_embeded, self.pvalue_embeded, self.shopping_embeded,
            self.occupation_embeded, self.new_user_class_embeded,
            self.campaign_embeded, self.customer_embeded, self.cate_embeded,
            self.brand_embeded
        ], -1)

        # sess_interest_division part
        #cate_sess_embeded = self.cate_embeddings_var(paddle.to_tensor(sess_input[:, ::2, :]))
        #brand_sess_embeded = self.brand_embeddings_var(paddle.to_tensor(sess_input[:, 1::2, :]))
        cate_sess_embeded = self.cate_embeddings_var(sess_input[:, ::2, :])
        brand_sess_embeded = self.brand_embeddings_var(sess_input[:, 1::2, :])

        # tr_input (n,5,10,8)
        tr_input = paddle.concat(
            [cate_sess_embeded, brand_sess_embeded], axis=-1)

        # sess interest extractor part
        lstm_input = []
        for i in range(self.sess_count):
            tr_sess_input = self.position_encoding(tr_input[:, i, :, :])
            tr_sess_input = self.transform(tr_sess_input)
            tr_sess_input = paddle.mean(tr_sess_input, axis=1, keepdim=True)
            lstm_input.append(tr_sess_input)

        lstm_input = paddle.concat(
            [
                lstm_input[0], lstm_input[1], lstm_input[2], lstm_input[3],
                lstm_input[4]
            ],
            axis=1)
        lstm_output, _ = self.bilstm(lstm_input)
        lstm_output = (lstm_output[:, :, :2 * self.sparse_embed_size] +
                       lstm_output[:, :, 2 * self.sparse_embed_size:]) / 2

        # sess interest activating layer
        lstm_input = self.transform_actpool(
            [query_embeded, lstm_input, sess_length])
        lstm_output = self.lstm_actpool(
            [query_embeded, lstm_output, sess_length])

        # concatenate all moudle output
        mlp_input = paddle.concat(
            [
                deep_input_embeded, paddle.nn.Flatten()(lstm_input),
                paddle.nn.Flatten()(lstm_output), self.price
            ],
            axis=-1)

        out = self.mlp(mlp_input)
        return out
