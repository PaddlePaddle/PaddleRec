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
import paddle.nn as nn
import paddle.nn.functional as F
import math
import numpy as np


class DMRLayer(nn.Layer):
    def __init__(self, user_size, cms_segid_size, cms_group_id_size,
                 final_gender_code_size, age_level_size, pvalue_level_size,
                 shopping_level_size, occupation_size,
                 new_user_class_level_size, adgroup_id_size, cate_size,
                 campaign_id_size, customer_size, brand_size, btag_size,
                 pid_size, main_embedding_size, other_embedding_size):
        super(DMRLayer, self).__init__()

        self.user_size = user_size
        self.cms_segid_size = cms_segid_size
        self.cms_group_id_size = cms_group_id_size
        self.final_gender_code_size = final_gender_code_size
        self.age_level_size = age_level_size
        self.pvalue_level_size = pvalue_level_size
        self.shopping_level_size = shopping_level_size
        self.occupation_size = occupation_size
        self.new_user_class_level_size = new_user_class_level_size
        self.adgroup_id_size = adgroup_id_size
        self.cate_size = cate_size
        self.campaign_id_size = campaign_id_size
        self.customer_size = customer_size
        self.brand_size = brand_size
        self.btag_size = btag_size
        self.pid_size = pid_size
        self.main_embedding_size = main_embedding_size
        self.other_embedding_size = other_embedding_size
        self.history_length = 50

        use_sparse = True
        if paddle.is_compiled_with_custom_device('npu'):
            use_sparse = False

        self.uid_embeddings_var = paddle.nn.Embedding(
            self.user_size,
            self.main_embedding_size,
            sparse=use_sparse,
            weight_attr=paddle.ParamAttr(
                name="UidSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.mid_embeddings_var = paddle.nn.Embedding(
            self.adgroup_id_size,
            self.main_embedding_size,
            sparse=use_sparse,
            weight_attr=paddle.ParamAttr(
                name="MidSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.cat_embeddings_var = paddle.nn.Embedding(
            self.cate_size,
            self.main_embedding_size,
            sparse=use_sparse,
            weight_attr=paddle.ParamAttr(
                name="CatSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.brand_embeddings_var = paddle.nn.Embedding(
            self.brand_size,
            self.main_embedding_size,
            sparse=use_sparse,
            weight_attr=paddle.ParamAttr(
                name="BrandSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.btag_embeddings_var = paddle.nn.Embedding(
            self.btag_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="BtagSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.dm_btag_embeddings_var = paddle.nn.Embedding(
            self.btag_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="DmBtagSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.campaign_id_embeddings_var = paddle.nn.Embedding(
            self.campaign_id_size,
            self.main_embedding_size,
            sparse=use_sparse,
            weight_attr=paddle.ParamAttr(
                name="CampSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.customer_embeddings_var = paddle.nn.Embedding(
            self.customer_size,
            self.main_embedding_size,
            sparse=use_sparse,
            weight_attr=paddle.ParamAttr(
                name="CustomSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.cms_segid_embeddings_var = paddle.nn.Embedding(
            self.cms_segid_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="CmsSegSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.cms_group_id_embeddings_var = paddle.nn.Embedding(
            self.cms_group_id_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="CmsGroupSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.final_gender_code_embeddings_var = paddle.nn.Embedding(
            self.final_gender_code_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="GenderSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.age_level_embeddings_var = paddle.nn.Embedding(
            self.age_level_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="AgeSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.pvalue_level_embeddings_var = paddle.nn.Embedding(
            self.pvalue_level_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="PvSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.shopping_level_embeddings_var = paddle.nn.Embedding(
            self.shopping_level_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="ShopSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.occupation_embeddings_var = paddle.nn.Embedding(
            self.occupation_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="OccupSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.new_user_class_level_embeddings_var = paddle.nn.Embedding(
            self.new_user_class_level_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="NewUserClsSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.pid_embeddings_var = paddle.nn.Embedding(
            self.pid_size,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="PidSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.position_his = paddle.arange(0, self.history_length)
        self.position_embeddings_var = paddle.nn.Embedding(
            self.history_length,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="PosSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.dm_position_his = paddle.arange(0, self.history_length)
        self.dm_position_embeddings_var = paddle.nn.Embedding(
            self.history_length,
            self.other_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="DmPosSparseFeatFactors",
                initializer=nn.initializer.Uniform()))
        self.query_layer = paddle.nn.Linear(
            self.other_embedding_size * 2,
            self.main_embedding_size * 2,
            name='dm_align')
        self.query_prelu = paddle.nn.PReLU(
            num_parameters=self.history_length, init=0.1, name='dm_prelu')
        self.att_layer1_layer = paddle.nn.Linear(
            self.main_embedding_size * 8, 80, name='dm_att_1')
        self.att_layer2_layer = paddle.nn.Linear(80, 40, name='dm_att_2')
        self.att_layer3_layer = paddle.nn.Linear(40, 1, name='dm_att_3')
        self.dnn_layer1_layer = paddle.nn.Linear(
            self.main_embedding_size * 2,
            self.main_embedding_size,
            name='dm_fcn_1')
        self.dnn_layer1_prelu = paddle.nn.PReLU(
            num_parameters=self.history_length, init=0.1, name='dm_fcn_1')

        self.query_layer2 = paddle.nn.Linear(
            (self.other_embedding_size + self.main_embedding_size) * 2,
            self.main_embedding_size * 2,
            name='dmr_align')
        self.query_prelu2 = paddle.nn.PReLU(
            num_parameters=self.history_length, init=0.1, name='dmr_prelu')
        self.att_layer1_layer2 = paddle.nn.Linear(
            self.main_embedding_size * 8, 80, name='tg_att_1')
        self.att_layer2_layer2 = paddle.nn.Linear(80, 40, name='tg_att_2')
        self.att_layer3_layer2 = paddle.nn.Linear(40, 1, name='tg_att_3')

        self.logits_layer = paddle.nn.Linear(self.main_embedding_size,
                                             self.cate_size)

        def deep_match(item_his_eb, context_his_eb, mask, match_mask,
                       mid_his_batch, item_vectors, item_biases, n_mid):
            query = context_his_eb
            query = self.query_layer(
                query)  # [-1, self.history_length, self.main_embedding_size*2]
            query = self.query_prelu(query)

            inputs = paddle.concat(
                [
                    query, item_his_eb, query - item_his_eb,
                    query * item_his_eb
                ],
                axis=-1)  # B,T,E
            att_layer1 = self.att_layer1_layer(inputs)
            att_layer1 = F.sigmoid(att_layer1)
            att_layer2 = self.att_layer2_layer(att_layer1)
            att_layer2 = F.sigmoid(att_layer2)
            att_layer3 = self.att_layer3_layer(att_layer2)  # B,T,1
            scores = paddle.transpose(att_layer3, [0, 2, 1])  # B,1,T

            # mask
            bool_mask = paddle.equal(mask, paddle.ones_like(mask))  # B,T
            key_masks = paddle.unsqueeze(bool_mask, axis=1)  # B,1,T
            paddings = paddle.ones_like(scores) * (-2**32 + 1)
            scores = paddle.where(key_masks, scores, paddings)

            # tril
            scores_tile = paddle.tile(
                paddle.sum(scores, axis=1),
                [1, paddle.shape(scores)[-1]])  # B, T*T
            scores_tile = paddle.reshape(scores_tile, [
                -1, paddle.shape(scores)[-1], paddle.shape(scores)[-1]
            ])  # B, T, T
            diag_vals = paddle.ones_like(scores_tile)  # B, T, T
            tril = paddle.tril(diag_vals)
            paddings = paddle.ones_like(tril) * (-2**32 + 1)
            scores_tile = paddle.where(
                paddle.equal(tril, paddle.full([1], 0.0, "float32")), paddings,
                scores_tile)  # B, T, T
            scores_tile = F.softmax(scores_tile)  # B, T, T

            att_dm_item_his_eb = paddle.matmul(scores_tile,
                                               item_his_eb)  # B, T, E
            dnn_layer1 = self.dnn_layer1_layer(att_dm_item_his_eb)
            dnn_layer1 = dnn_layer1.reshape(
                [-1, self.history_length, self.main_embedding_size])  ##
            dnn_layer1 = self.dnn_layer1_prelu(dnn_layer1)

            # target mask
            user_vector = dnn_layer1[:, -1, :]  # B, E
            user_vector2 = dnn_layer1[:, -2, :] * paddle.reshape(
                match_mask,
                [-1, paddle.shape(match_mask)[1], 1])[:, -2, :]  # B, E

            num_sampled = 2000
            labels = paddle.reshape(mid_his_batch[:, -1], [-1, 1])  # B, 1

            # not sample, slow
            # [B, E] * [E_size, cate_size]
            logits = paddle.matmul(
                user_vector2, item_vectors, transpose_y=True)
            logits = paddle.add(logits, item_biases)
            loss = F.cross_entropy(input=logits, label=labels)

            return loss, user_vector, scores

        def dmr_fcn_attention(item_eb,
                              item_his_eb,
                              context_his_eb,
                              mask,
                              mode='SUM'):
            mask = paddle.equal(mask, paddle.ones_like(mask))
            item_eb_tile = paddle.tile(item_eb,
                                       [1, paddle.shape(mask)[1]])  # B, T*E
            item_eb_tile = paddle.reshape(
                item_eb_tile,
                [-1, paddle.shape(mask)[1], item_eb.shape[-1]])  # B, T, E
            if context_his_eb is None:
                query = item_eb_tile
            else:
                query = paddle.concat([item_eb_tile, context_his_eb], axis=-1)
            query = self.query_layer2(query)
            query = self.query_prelu2(query)
            dmr_all = paddle.concat(
                [
                    query, item_his_eb, query - item_his_eb,
                    query * item_his_eb
                ],
                axis=-1)
            att_layer_1 = self.att_layer1_layer2(dmr_all)
            att_layer_1 = F.sigmoid(att_layer_1)
            att_layer_2 = self.att_layer2_layer2(att_layer_1)
            att_layer_2 = F.sigmoid(att_layer_2)
            att_layer_3 = self.att_layer3_layer2(att_layer_2)  # B, T, 1
            att_layer_3 = paddle.reshape(
                att_layer_3, [-1, 1, paddle.shape(item_his_eb)[1]])  # B,1,T
            scores = att_layer_3
            scores = scores.reshape([-1, 1, self.history_length])  ##

            # Mask
            key_masks = paddle.unsqueeze(mask, 1)  # B,1,T
            paddings = paddle.ones_like(scores) * (-2**32 + 1)
            paddings_no_softmax = paddle.zeros_like(scores)
            scores = paddle.where(key_masks, scores, paddings)  # [B, 1, T]
            scores_no_softmax = paddle.where(key_masks, scores,
                                             paddings_no_softmax)

            scores = F.softmax(scores)

            if mode == 'SUM':
                output = paddle.matmul(scores, item_his_eb)  # [B, 1, H]
                output = paddle.sum(output, axis=1)  # B,E
            else:
                scores = paddle.reshape(scores,
                                        [-1, paddle.shape(item_his_eb)[1]])
                output = item_his_eb * paddle.unsqueeze(scores, -1)
                output = paddle.reshape(output, paddle.shape(item_his_eb))

            return output, scores, scores_no_softmax

        self._deep_match = deep_match
        self._dmr_fcn_attention = dmr_fcn_attention

        self.dm_item_vectors_var = paddle.nn.Embedding(
            self.cate_size,
            self.main_embedding_size,
            # sparse=True,
            weight_attr=paddle.ParamAttr(
                name="DmItemSparseFeatFactors",
                initializer=nn.initializer.Uniform()))

        self.dm_item_biases = paddle.zeros(
            shape=[self.cate_size], dtype='float32')

        self.inp_length = self.main_embedding_size + (
            self.other_embedding_size * 8 + self.main_embedding_size * 5 + 1 +
            self.other_embedding_size + self.main_embedding_size * 2 +
            self.main_embedding_size * 2 + 1 + 1 + self.main_embedding_size * 2
        )
        self.inp_layer = paddle.nn.BatchNorm(
            self.inp_length, momentum=0.99, epsilon=1e-03)
        self.dnn0_layer = paddle.nn.Linear(self.inp_length, 512, name='f0')
        self.dnn0_prelu = paddle.nn.PReLU(
            num_parameters=512, init=0.1, name='prelu0')
        self.dnn1_layer = paddle.nn.Linear(512, 256, name='f1')
        self.dnn1_prelu = paddle.nn.PReLU(
            num_parameters=256, init=0.1, name='prelu1')
        self.dnn2_layer = paddle.nn.Linear(256, 128, name='f2')
        self.dnn2_prelu = paddle.nn.PReLU(
            num_parameters=128, init=0.1, name='prelu2')
        self.dnn3_layer = paddle.nn.Linear(128, 1, name='f3')
        self.dnn3_prelu = paddle.nn.PReLU(
            num_parameters=1, init=0.1, name='prelu3')

    def forward(self, inputs_tensor, is_infer=0):
        # input
        inputs = inputs_tensor[0]  # sparse_tensor
        dense_tensor = inputs_tensor[1]
        self.btag_his = inputs[:, 0:self.history_length]
        self.cate_his = inputs[:, self.history_length:self.history_length * 2]
        self.brand_his = inputs[:, self.history_length * 2:self.history_length
                                * 3]
        self.mask = inputs[:, self.history_length * 3:self.history_length * 4]
        self.match_mask = inputs[:, self.history_length * 4:self.history_length
                                 * 5]

        self.uid = inputs[:, self.history_length * 5]
        self.cms_segid = inputs[:, self.history_length * 5 + 1]
        self.cms_group_id = inputs[:, self.history_length * 5 + 2]
        self.final_gender_code = inputs[:, self.history_length * 5 + 3]
        self.age_level = inputs[:, self.history_length * 5 + 4]
        self.pvalue_level = inputs[:, self.history_length * 5 + 5]
        self.shopping_level = inputs[:, self.history_length * 5 + 6]
        self.occupation = inputs[:, self.history_length * 5 + 7]
        self.new_user_class_level = inputs[:, self.history_length * 5 + 8]

        self.mid = inputs[:, self.history_length * 5 + 9]
        self.cate_id = inputs[:, self.history_length * 5 + 10]
        self.campaign_id = inputs[:, self.history_length * 5 + 11]
        self.customer = inputs[:, self.history_length * 5 + 12]
        self.brand = inputs[:, self.history_length * 5 + 13]
        self.price = dense_tensor.astype('float32')

        self.pid = inputs[:, self.history_length * 5 + 15]

        if is_infer == 0:
            self.labels = inputs[:, self.history_length * 5 + 16]

        # embedding layer
        self.uid_batch_embedded = self.uid_embeddings_var(self.uid)
        self.mid_batch_embedded = self.mid_embeddings_var(self.mid)
        self.cat_batch_embedded = self.cat_embeddings_var(self.cate_id)
        self.cat_his_batch_embedded = self.cat_embeddings_var(self.cate_his)
        self.brand_batch_embedded = self.brand_embeddings_var(self.brand)
        self.brand_his_batch_embedded = self.brand_embeddings_var(
            self.brand_his)
        self.btag_his_batch_embedded = self.btag_embeddings_var(self.btag_his)
        self.dm_btag_his_batch_embedded = self.dm_btag_embeddings_var(
            self.btag_his)
        self.campaign_id_batch_embedded = self.campaign_id_embeddings_var(
            self.campaign_id)
        self.customer_batch_embedded = self.customer_embeddings_var(
            self.customer)
        self.cms_segid_batch_embedded = self.cms_segid_embeddings_var(
            self.cms_segid)
        self.cms_group_id_batch_embedded = self.cms_group_id_embeddings_var(
            self.cms_group_id)
        self.final_gender_code_batch_embedded = self.final_gender_code_embeddings_var(
            self.final_gender_code)
        self.age_level_batch_embedded = self.age_level_embeddings_var(
            self.age_level)
        self.pvalue_level_batch_embedded = self.pvalue_level_embeddings_var(
            self.pvalue_level)
        self.shopping_level_batch_embedded = self.shopping_level_embeddings_var(
            self.shopping_level)
        self.occupation_batch_embedded = self.occupation_embeddings_var(
            self.occupation)
        self.new_user_class_level_batch_embedded = self.new_user_class_level_embeddings_var(
            self.new_user_class_level)
        self.pid_batch_embedded = self.pid_embeddings_var(self.pid)

        self.user_feat = paddle.concat([
            self.uid_batch_embedded, self.cms_segid_batch_embedded,
            self.cms_group_id_batch_embedded,
            self.final_gender_code_batch_embedded,
            self.age_level_batch_embedded, self.pvalue_level_batch_embedded,
            self.shopping_level_batch_embedded, self.occupation_batch_embedded,
            self.new_user_class_level_batch_embedded
        ], -1)
        self.item_his_eb = paddle.concat(
            [self.cat_his_batch_embedded, self.brand_his_batch_embedded], -1)

        self.item_his_eb_sum = paddle.sum(self.item_his_eb, 1)
        self.item_feat = paddle.concat([
            self.mid_batch_embedded, self.cat_batch_embedded,
            self.brand_batch_embedded, self.campaign_id_batch_embedded,
            self.customer_batch_embedded, self.price
        ], -1)
        self.item_eb = paddle.concat(
            [self.cat_batch_embedded, self.brand_batch_embedded], -1)
        self.context_feat = self.pid_batch_embedded

        self.position_his_eb = self.position_embeddings_var(
            self.position_his)  # T, E
        self.position_his_eb = paddle.tile(
            self.position_his_eb, [paddle.shape(self.mid)[0], 1])  # B*T, E
        self.position_his_eb = paddle.reshape(self.position_his_eb, [
            paddle.shape(self.mid)[0], -1,
            paddle.shape(self.position_his_eb)[1]
        ])  # B, T, E

        self.dm_position_his_eb = self.dm_position_embeddings_var(
            self.dm_position_his)  # T, E
        self.dm_position_his_eb = paddle.tile(
            self.dm_position_his_eb, [paddle.shape(self.mid)[0], 1])  # B*T, E
        self.dm_position_his_eb = paddle.reshape(self.dm_position_his_eb, [
            paddle.shape(self.mid)[0], -1,
            paddle.shape(self.dm_position_his_eb)[1]
        ])  # B, T, E

        self.position_his_eb = paddle.concat(
            [self.position_his_eb, self.btag_his_batch_embedded], -1)
        self.dm_position_his_eb = paddle.concat(
            [self.dm_position_his_eb, self.dm_btag_his_batch_embedded], -1)

        # User-to-Item Network
        # Auxiliary Match Network
        self.match_mask = paddle.cast(self.match_mask, 'float32')
        self.aux_loss, self.dm_user_vector, scores = self._deep_match(
            self.item_his_eb, self.dm_position_his_eb, self.mask,
            self.match_mask, self.cate_his, self.dm_item_vectors_var.weight,
            self.dm_item_biases, self.cate_size)
        self.aux_loss *= 0.1
        self.dm_item_vec = self.dm_item_vectors_var(self.cate_id)
        rel_u2i = paddle.sum(self.dm_user_vector * self.dm_item_vec,
                             -1,
                             keepdim=True)  # B,1
        self.rel_u2i = rel_u2i

        # Item-to-Item Network
        att_outputs, alphas, scores_unnorm = self._dmr_fcn_attention(
            self.item_eb, self.item_his_eb, self.position_his_eb, self.mask)
        rel_i2i = paddle.unsqueeze(paddle.sum(scores_unnorm, [1, 2]), -1)
        self.rel_i2i = rel_i2i
        self.scores = paddle.sum(alphas, 1)
        inp = paddle.concat([
            self.user_feat, self.item_feat, self.context_feat,
            self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, rel_u2i,
            rel_i2i, att_outputs
        ], -1)

        # build fcn net
        inp = self.inp_layer(inp)
        dnn0 = self.dnn0_layer(inp)
        dnn0 = self.dnn0_prelu(dnn0)
        dnn1 = self.dnn1_layer(dnn0)
        dnn1 = self.dnn1_prelu(dnn1)
        dnn2 = self.dnn2_layer(dnn1)
        dnn2 = self.dnn2_prelu(dnn2)
        dnn3 = self.dnn3_layer(dnn2)
        dnn3 = self.dnn3_prelu(dnn3)

        # prediction
        self.y_hat = F.sigmoid(dnn3)

        if is_infer == False:
            # Cross-entropy loss and optimizer initialization
            x = paddle.sum(dnn3, 1)
            BCE = paddle.nn.BCEWithLogitsLoss()
            ctr_loss = paddle.mean(BCE(x, label=self.labels.astype('float32')))
            self.ctr_loss = ctr_loss
            self.loss = self.ctr_loss + self.aux_loss

            return self.y_hat, self.loss
        else:
            return self.y_hat, paddle.ones(shape=[1])
