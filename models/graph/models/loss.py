#-*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
"""
    doc
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import math
import time
import numpy as np
import pickle as pkl

import paddle
import paddle.incubate as F
import pgl
from pgl.utils.logger import log
import paddle.static as static


def hinge_loss(config, predictions):
    """doc
    """
    log.info("use hinge loss")
    logits = predictions["logits"]
    log.info("using hinge loss")
    pos = logits[:, 0:1]
    neg = logits[:, 1:]
    loss = paddle.sum(paddle.nn.functional.relu(neg - pos + config.margin))

    return loss


def nce_loss(config, predictions):
    """doc
    """
    log.info("use nce loss")
    logits = predictions["logits"]
    tao = 5
    # equal to L.elementwise_div(logits, 1/tao) and it will boardcast automatically
    logits = logits * tao
    labels = paddle.zeros([paddle.shape(logits)[0], 1], dtype="int64")
    loss = paddle.nn.functional.softmax_with_cross_entropy(logits, labels)
    loss = paddle.sum(loss)
    return loss


def hcl_loss(config, hcl_logits_list):
    """ hierarchical contrastive learning loss"""
    log.info("use hcl loss")
    tao = 5
    hcl_loss = 0.0
    for logits in hcl_logits_list:
        # equal to L.elementwise_div(logits, 1/tao) and it will boardcast automatically
        logits = logits * tao
        labels = paddle.zeros([paddle.shape(logits)[0], 1], dtype="int64")
        loss = paddle.nn.functional.softmax_with_cross_entropy(logits, labels)
        loss = paddle.sum(loss)
        hcl_loss += loss

    return hcl_loss / len(hcl_logits_list)


def sigmoid_loss(config, predictions):
    """doc
    """
    log.info("use sigmoid loss")
    logits = paddle.unsqueeze(predictions["logits"], axis=1)

    pos_label = paddle.full(shape=[paddle.shape(logits)[0], 1, 1],
                             dtype="float32", fill_value=1.0)

    neg_label = paddle.full(shape=[paddle.shape(logits)[0], 1, config.neg_num],
                             dtype="float32", fill_value=0.0)

    label = paddle.concat([pos_label, neg_label], -1)


    pos_weight = paddle.full(shape=[paddle.shape(logits)[0], 1, 1],
                             dtype="float32", fill_value=config.neg_num)

    neg_weight = paddle.full(shape=[paddle.shape(logits)[0], 1, 1],
                             dtype="float32", fill_value=1.0)

    weight = paddle.concat([pos_weight, neg_weight], -1)

    weight.stop_gradient = True
    label.stop_gradient = True

    loss = paddle.nn.functional.binary_cross_entropy_with_logits(logits, label)
    loss = loss * weight
    loss = paddle.sum(loss)

    return loss


def simgcl_loss(config, predictions):
    """doc
    """
    log.info("use simgcl loss")
    nfeat = predictions["nfeat"]       # nfeat: [batch_size, 2, hidden_size]
    src_feat = predictions["src_nfeat"]

    noise = paddle.uniform(paddle.shape(src_feat), dtype="float32", min=0.0, max=1.0)
    noise = noise * paddle.sign(src_feat)
    ratio = 0.5
    noised_nfeat = src_feat + noise * ratio

    neighbor_dsts_feat_all = [noised_nfeat]

    for neg in range(config.neg_num):
        neighbor_dsts_feat_all.append(
            F.layers.shuffle_batch(neighbor_dsts_feat_all[0]))
    neighbor_dsts_feat = paddle.concat(neighbor_dsts_feat_all, axis=1)

    noised_logits = paddle.matmul(src_feat, neighbor_dsts_feat, transpose_y=True)
    # noised_logits [batch_size, 1]

    tao = 5 
    # equal to L.elementwise_div(logits, 1/tao) and it will boardcast automatically
    noised_logits = noised_logits * tao
    noised_logits = paddle.squeeze(noised_logits, [1])
    labels = paddle.zeros([paddle.shape(noised_logits)[0], 1], dtype="int64")
    loss = paddle.nn.functional.softmax_with_cross_entropy(noised_logits, labels)
    loss = paddle.sum(loss)
    return loss


def in_batch_negative_softmax_loss(config, predictions):
    """doc
    """
    logits = predictions["logits"]
    if config.liwb_debug:
        probs = paddle.nn.functional.softmax(logits)
        static.Print(logits, message="logits", summarize=64)
        static.Print(probs, message="probs", summarize=64)
    labels = paddle.unsqueeze(paddle.arange(0, paddle.shape(logits)[0], dtype="int64"), axis=1)
    loss = paddle.nn.functional.softmax_with_cross_entropy(logits, labels)
    loss = paddle.mean(loss)
    return loss

def graph_negative_softmax_loss(config, predictions):
    """doc
    """

    def _gns(center_src, neigh_src, center_dst, neigh_dst):
        bz, seq_len_b, dim = paddle.shape(neigh_dst)

        pos_score = paddle.sum(center_src * center_dst, axis=1, keepdim=True)
        # pos_score (bz, 1)

        neigh_dst = paddle.reshape(neigh_dst, [bz * seq_len_b, dim])
        # neigh_dst (bz * seq_len_b, 128)

        # center_src (bz, 128)
        mat_b = paddle.matmul(center_src, neigh_dst, transpose_y=True)
        # mat_b (bz, bz * seq_len_b)

        mat_b = paddle.reshape(mat_b, shape=[bz, bz, seq_len_b])
        # (bz, bz, seq_len_b)

        softmax_margin = paddle.unsqueeze(paddle.eye(paddle.shape(mat_b)[0]) * 9999, axis=2)
        # (bz, bz, eye*9999)
        # (bz, bz, seq_len_b)
        mat_b = mat_b - softmax_margin
        # (bz, bz * seq_len_b)
        mat_b = paddle.reshape(mat_b, shape=[bz, bz * seq_len_b])
        # (bz, bz * seq_len_b + 1)
        mat_b = paddle.concat([pos_score, mat_b], axis=1)

        labels = paddle.zeros([paddle.shape(mat_b)[0], 1], dtype="int64")
        loss = paddle.mean(paddle.nn.functional.softmax_with_cross_entropy(mat_b, labels))
        return loss

    nfeat = predictions["nfeat"]
    all_nfeat = predictions["all_nfeat"]
    noise_nfeat = predictions["noise_nfeat"]
    noise_all_nfeat = predictions["noise_all_nfeat"]

    src, dst = nfeat[:, 0, :], nfeat[:, 1, :]
    neigh_src, neigh_dst = all_nfeat[:, 0, :, :], all_nfeat[:, 1, :, :]

    noise_src, noise_dst = noise_nfeat[:, 0, :], noise_nfeat[:, 1, :]
    noise_neigh_src, noise_neigh_dst = noise_all_nfeat[:, 0, :, :], noise_all_nfeat[:, 1, :, :]

    loss1 = _gns(src, neigh_src, dst, neigh_dst)
    loss2 = _gns(noise_src, noise_neigh_src, noise_dst, noise_neigh_dst)
    loss3 = _gns(src, neigh_src, noise_dst, noise_neigh_dst)
    loss4 = _gns(noise_src, noise_neigh_src, dst, neigh_dst)
    loss = loss1 + loss2 + loss3 + loss4
    return loss