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
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from lastfm_reader import RecDataset


class MHCN(nn.Layer):
    def __init__(self, n_layers=2, emb_size=50, config=None):
        super(MHCN, self).__init__()
        self.n_layers = n_layers
        self.emb_size = emb_size
        self.config = config
        self.ss_rate = 0.01

        self.recDataset = RecDataset(file_list="data/train", config=config)
        self.data, self.social = self.recDataset.get_dataset()
        self.num_users, self.num_items, _ = self.data.trainingSize()

        self.userAdjacency = None
        self.itemAdjacency = None

        self.H_s, self.H_j, self.H_p = self.buildMotifInducedAdjacencyMatrix()
        self.H_s = paddle.to_tensor(self.H_s, dtype="float32")
        self.H_j = paddle.to_tensor(self.H_j, dtype="float32")
        self.H_p = paddle.to_tensor(self.H_p, dtype="float32")
        self.R = paddle.to_tensor(self.buildJointAdjacency(), dtype="float32")

        self.weights = {}

        self.n_channel = 4
        # define learnable paramters
        for i in range(self.n_channel):
            name = "gating%d" % (i + 1)
            self.weights[name] = paddle.create_parameter(
                name=name,
                shape=[self.emb_size, self.emb_size],
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform(),
                    regularizer=paddle.regularizer.L2Decay(coeff=0.001)),
                dtype="float32")
            self.add_parameter(name, self.weights[name])

            name = "gating_bias%d" % (i + 1)
            self.weights[name] = paddle.create_parameter(
                name=name,
                shape=[1, self.emb_size],
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform(),
                    regularizer=paddle.regularizer.L2Decay(coeff=0.001)),
                dtype="float32")
            self.add_parameter(name, self.weights[name])

            name = "sgating%d" % (i + 1)
            self.weights[name] = paddle.create_parameter(
                name=name,
                shape=[self.emb_size, self.emb_size],
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform(),
                    regularizer=paddle.regularizer.L2Decay(coeff=0.001)),
                dtype="float32")
            self.add_parameter(name, self.weights[name])

            name = "sgating_bias%d" % (i + 1)
            self.weights[name] = paddle.create_parameter(
                name=name,
                shape=[1, self.emb_size],
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform(),
                    regularizer=paddle.regularizer.L2Decay(coeff=0.001)),
                dtype="float32")
            self.add_parameter(name, self.weights[name])

        self.weights["attention"] = paddle.create_parameter(
            name="attention",
            shape=[1, self.emb_size],
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform(),
                regularizer=paddle.regularizer.L2Decay(coeff=0.001)),
            dtype="float32")
        self.add_parameter("attention", self.weights["attention"])

        self.weights["attention_mat"] = paddle.create_parameter(
            name="attention_mat",
            shape=[self.emb_size, self.emb_size],
            dtype="float32",
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform(),
                regularizer=paddle.regularizer.L2Decay(coeff=0.001)))
        self.add_parameter("attention_mat", self.weights["attention_mat"])

        self.user_embeddings = paddle.nn.Embedding(
            name="user_embeddings",
            num_embeddings=self.num_users,
            embedding_dim=self.emb_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.005),
                regularizer=paddle.regularizer.L2Decay(coeff=0.001)))

        self.item_embeddings = paddle.nn.Embedding(
            name="item_embeddings",
            num_embeddings=self.num_items,
            embedding_dim=self.emb_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(std=0.005),
                regularizer=paddle.regularizer.L2Decay(coeff=0.001)))

    # user-user
    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0]
        AdjacencyMatrix = coo_matrix(
            (entries, (row, col)),
            shape=(self.num_users, self.num_users),
            dtype=np.float32)
        return AdjacencyMatrix

    # user-item
    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0]
        ratingMatrix = coo_matrix(
            (entries, (row, col)),
            shape=(self.num_users, self.num_items),
            dtype=np.float32)
        return ratingMatrix

    def buildJointAdjacency(self):
        indices = np.array(
            [[self.data.user[item[0]], self.data.item[item[1]]]
             for item in self.data.trainingData])
        values = np.array([
            float(item[2]) / math.sqrt(len(self.data.trainSet_u[item[0]])) /
            math.sqrt(len(self.data.trainSet_i[item[1]]))
            for item in self.data.trainingData
        ])

        norm_adj = coo_matrix(
            (values, (indices[:, 0], indices[:, 1])),
            shape=(self.num_users, self.num_items)).toarray()

        return norm_adj

    def buildMotifInducedAdjacencyMatrix(self):
        # social graph, user-user
        S = self.buildSparseRelationMatrix()
        Y = self.buildSparseRatingMatrix()
        self.userAdjacency = Y.tocsr()
        self.itemAdjacency = Y.T.tocsr()
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (
            U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)
                                                                ).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)
                                                                  ).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (
            U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (
            U.dot(U.T)).multiply(B)
        A8 = (Y.dot(Y.T)).multiply(B)
        A9 = (Y.dot(Y.T)).multiply(U)
        A9 = A9 + A9.T
        A10 = Y.dot(Y.T) - A8 - A9
        # addition and row-normalization
        H_s = sum([A1, A2, A3, A4, A5, A6, A7])
        H_s = H_s.multiply(1.0 / H_s.sum(axis=1).reshape(-1, 1))
        H_j = sum([A8, A9])
        H_j = H_j.multiply(1.0 / H_j.sum(axis=1).reshape(-1, 1))
        H_p = A10
        H_p = H_p.multiply(H_p > 1)
        H_p = H_p.multiply(1.0 / H_p.sum(axis=1).reshape(-1, 1))

        return [H_s.toarray(), H_j.toarray(), H_p.toarray()]

    def self_gating(self, em, channel):
        """
        em: (num_users, emb_size)
        """
        return paddle.multiply(
            em,
            F.sigmoid(
                paddle.matmul(em, self.weights["gating%d" % channel]) +
                self.weights["gating_bias%d" % channel]))

    def self_supervised_gating(self, em, channel):
        return paddle.multiply(
            em,
            F.sigmoid(
                paddle.matmul(em, self.weights["sgating%d" % channel]) +
                self.weights["sgating_bias%d" % channel]))

    def channel_attention(self, *channel_embeddings):
        """
            channel_embeddings_1: (num_user, emb_size)
            attention_mat: (emb_size, emb_size)
            attention: (1, emb_size)
        """
        weights = []
        for embedding in channel_embeddings:
            # ((num_user, emb_size) * (emb_size, emb_size)) @ (1, emb_size) = (num_user, emb_size) @ (1, emb_size)
            # = (num_user, emb_size) -> (num_user, )
            weights.append(
                paddle.sum(
                    paddle.multiply(
                        paddle.matmul(embedding, self.weights[
                            "attention_mat"]), self.weights["attention"]), 1))
        t = paddle.stack(weights)
        # (num_user, channel_num)
        score = F.softmax(paddle.transpose(t, perm=[1, 0]))
        mixed_embeddings = 0.0
        for i in range(len(weights)):
            # (emb_size, num_user) @
            # (num_user, emb_size) @ (num_user, 1) -> (num_user, emb_size)
            mixed_embeddings += paddle.transpose(
                paddle.multiply(
                    paddle.transpose(
                        channel_embeddings[i], perm=[1, 0]),
                    paddle.transpose(
                        score, perm=[1, 0])[i]),
                perm=[1, 0])
        return mixed_embeddings, score

    def infer_embedding(self):

        # self-gating
        user_embeddings_c1 = self.self_gating(self.user_embeddings.weight, 1)
        user_embeddings_c2 = self.self_gating(self.user_embeddings.weight, 2)
        user_embeddings_c3 = self.self_gating(self.user_embeddings.weight, 3)
        simple_user_embeddings = self.self_gating(self.user_embeddings.weight,
                                                  4)
        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        item_embeddings = self.item_embeddings.weight
        all_embeddings_i = [item_embeddings]

        # multi-channel convolution
        for k in range(self.n_layers):
            mixed_embedding = self.channel_attention(
                user_embeddings_c1, user_embeddings_c2,
                user_embeddings_c3)[0] + simple_user_embeddings / 2.0
            # Channel S
            user_embeddings_c1 = paddle.matmul(self.H_s, user_embeddings_c1)
            norm_embeddings = F.normalize(user_embeddings_c1, axis=1, p=2)
            all_embeddings_c1 += [norm_embeddings]
            # Channel J
            user_embeddings_c2 = paddle.matmul(self.H_j, user_embeddings_c2)
            norm_embeddings = F.normalize(user_embeddings_c2, axis=1, p=2)
            all_embeddings_c2 += [norm_embeddings]
            # Channel P
            user_embeddings_c3 = paddle.matmul(self.H_p, user_embeddings_c3)
            norm_embeddings = F.normalize(user_embeddings_c3, axis=1, p=2)
            all_embeddings_c3 += [norm_embeddings]
            # item convolution
            new_item_embeddings = paddle.matmul(
                paddle.transpose(
                    self.R, perm=[1, 0]), mixed_embedding)
            norm_embeddings = F.normalize(new_item_embeddings, axis=1, p=2)
            all_embeddings_i += [norm_embeddings]
            simple_user_embeddings = paddle.matmul(self.R, item_embeddings)
            all_embeddings_simple += [
                F.normalize(
                    simple_user_embeddings, axis=1, p=2)
            ]
            item_embeddings = new_item_embeddings

        # averaging the channel-specific embeddings
        user_embeddings_c1 = paddle.sum(paddle.stack(all_embeddings_c1),
                                        axis=0)
        user_embeddings_c2 = paddle.sum(paddle.stack(all_embeddings_c2),
                                        axis=0)
        user_embeddings_c3 = paddle.sum(paddle.stack(all_embeddings_c3),
                                        axis=0)
        simple_user_embeddings = paddle.sum(
            paddle.stack(all_embeddings_simple), axis=0)
        item_embeddings = paddle.sum(paddle.stack(all_embeddings_i), axis=0)

        # aggregating channel-specific embeddings
        final_item_embeddings = item_embeddings
        final_user_embeddings, attention_score = self.channel_attention(
            user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)
        final_user_embeddings += simple_user_embeddings / 2

        return final_user_embeddings, final_item_embeddings

    def forward(self, inputs):
        """
        u_idx, v_idx, neg_idx = inputs
        """
        u_idx, v_idx, neg_idx = inputs

        final_user_embeddings, final_item_embeddings = self.infer_embedding()

        # create self-supervised loss
        ss_loss = 0.0
        ss_loss += self.hierarchical_self_supervision(
            self.self_supervised_gating(final_user_embeddings, 1), self.H_s)
        ss_loss += self.hierarchical_self_supervision(
            self.self_supervised_gating(final_user_embeddings, 2), self.H_j)
        ss_loss += self.hierarchical_self_supervision(
            self.self_supervised_gating(final_user_embeddings, 3), self.H_p)

        # embedding look-up
        batch_neg_item_emb = F.embedding(
            weight=final_item_embeddings, x=neg_idx)
        batch_user_emb = F.embedding(weight=final_user_embeddings, x=u_idx)
        batch_pos_item_emb = F.embedding(weight=final_item_embeddings, x=v_idx)

        return batch_user_emb, batch_pos_item_emb, batch_neg_item_emb, ss_loss

    def hierarchical_self_supervision(self, em, adj):
        def row_shuffle(embedding):
            return embedding[paddle.randperm(paddle.shape(embedding)[0])]

        def row_column_shuffle(embedding):
            embedding = paddle.transpose(embedding, perm=[1, 0])
            corrupted_embedding = paddle.transpose(
                embedding[paddle.randperm(paddle.shape(embedding)[0])],
                perm=[1, 0])
            return corrupted_embedding[paddle.randperm(
                paddle.shape(corrupted_embedding)[0])]

        def score(x1, x2):
            return paddle.sum(paddle.multiply(x1, x2), axis=1)

        user_embeddings = em
        edge_embeddings = paddle.matmul(adj, user_embeddings)

        # Local MIN
        pos = score(user_embeddings, edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        local_loss = paddle.sum(-paddle.log(F.sigmoid(pos - neg1)) -
                                paddle.log(F.sigmoid(neg1 - neg2)))

        # Global MIN
        graph = paddle.mean(edge_embeddings, axis=0)
        pos = score(edge_embeddings, graph)
        neg1 = score(row_column_shuffle(edge_embeddings), graph)
        global_loss = paddle.sum(-paddle.log(F.sigmoid(pos - neg1)))

        return global_loss + local_loss
