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

    pos_label = paddle.full(
        shape=[paddle.shape(logits)[0], 1, 1], dtype="float32", fill_value=1.0)

    neg_label = paddle.full(
        shape=[paddle.shape(logits)[0], 1, config.neg_num],
        dtype="float32",
        fill_value=0.0)

    label = paddle.concat([pos_label, neg_label], -1)

    pos_weight = paddle.full(
        shape=[paddle.shape(logits)[0], 1, 1],
        dtype="float32",
        fill_value=config.neg_num)

    neg_weight = paddle.full(
        shape=[paddle.shape(logits)[0], 1, 1], dtype="float32", fill_value=1.0)

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
    nfeat = predictions["nfeat"]
    src_feat = predictions["src_nfeat"]

    noise = paddle.uniform(
        paddle.shape(nfeat), dtype="float32", min=0.0, max=1.0)
    noise = noise * paddle.sign(nfeat)
    ratio = 0.5
    noised_nfeat = nfeat + noise * ratio
    noised_src_nfeat = noised_nfeat[:, 0:1, :]

    noised_logits = paddle.matmul(src_feat, noised_src_nfeat, transpose_y=True)
    noised_logits = paddle.squeeze(noised_logits, axis=[1])

    tao = 100
    # equal to L.elementwise_div(logits, 1/tao) and it will boardcast automatically
    noised_logits = noised_logits * tao
    labels = paddle.zeros([paddle.shape(noised_logits)[0], 1], dtype="int64")
    loss = paddle.nn.functional.softmax_with_cross_entropy(noised_logits,
                                                           labels)
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
    labels = paddle.unsqueeze(
        paddle.arange(
            0, paddle.shape(logits)[0], dtype="int64"), axis=1)
    loss = paddle.nn.functional.softmax_with_cross_entropy(logits, labels)
    loss = paddle.mean(loss)
    return loss
