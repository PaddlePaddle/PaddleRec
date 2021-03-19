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
import numpy as np


class Mind_SampledSoftmaxLoss_Layer(nn.Layer):
    """SampledSoftmaxLoss with LogUniformSampler
    """

    def __init__(self,
                 num_classes,
                 n_sample,
                 unique=True,
                 remove_accidental_hits=True,
                 subtract_log_q=True,
                 num_true=1,
                 batch_size=None):
        super(Mind_SampledSoftmaxLoss_Layer, self).__init__()
        self.range_max = num_classes
        self.n_sample = n_sample
        self.unique = unique
        self.remove_accidental_hits = remove_accidental_hits
        self.subtract_log_q = subtract_log_q
        self.num_true = num_true
        self.prob = np.array([0.0] * self.range_max)
        self.batch_size = batch_size
        for i in range(1, self.range_max):
            self.prob[i] = (np.log(i+2) - np.log(i+1)) / \
                np.log(self.range_max + 1)
        self.new_prob = paddle.assign(self.prob.astype("float32"))
        self.log_q = paddle.log(-(paddle.exp((-paddle.log1p(self.new_prob) * 2
                                              * n_sample)) - 1.0))

    def sample(self, labels):
        """Random sample neg_samples
        """
        n_sample = self.n_sample
        n_tries = 1 * n_sample
        neg_samples = paddle.multinomial(
            self.new_prob,
            num_samples=n_sample,
            replacement=self.unique is False)
        true_log_probs = paddle.gather(self.log_q, labels)
        samp_log_probs = paddle.gather(self.log_q, neg_samples)
        return true_log_probs, samp_log_probs, neg_samples

    def forward(self, inputs, labels, weights, bias):
        """forward
        """
        # weights.stop_gradient = False
        embedding_dim = paddle.shape(weights)[-1]
        true_log_probs, samp_log_probs, neg_samples = self.sample(labels)
        n_sample = neg_samples.shape[0]

        b1 = paddle.shape(labels)[0]
        b2 = paddle.shape(labels)[1]

        all_ids = paddle.concat([labels.reshape((-1, )), neg_samples])
        all_w = paddle.gather(weights, all_ids)

        true_w = all_w[:-n_sample].reshape((-1, b2, embedding_dim))
        sample_w = all_w[-n_sample:].reshape((n_sample, embedding_dim))

        all_b = paddle.gather(bias, all_ids)
        true_b = all_b[:-n_sample].reshape((-1, 1))

        sample_b = all_b[-n_sample:]

        # [B, D] * [B, 1,D]
        true_logist = paddle.matmul(
            true_w, inputs.unsqueeze(1), transpose_y=True).squeeze(1) + true_b

        sample_logist = paddle.matmul(
            inputs.unsqueeze(1), sample_w, transpose_y=True) + sample_b

        if self.subtract_log_q:
            true_logist = true_logist - true_log_probs.unsqueeze(1)
            sample_logist = sample_logist - samp_log_probs

        if self.remove_accidental_hits:
            hit = (paddle.equal(labels[:, :], neg_samples)).unsqueeze(1)
            padding = paddle.ones_like(sample_logist) * -1e30
            sample_logist = paddle.where(hit, padding, sample_logist)

        sample_logist = sample_logist.squeeze(1)
        out_logist = paddle.concat([true_logist, sample_logist], axis=1)
        out_label = paddle.concat(
            [
                paddle.ones_like(true_logist) / self.num_true,
                paddle.zeros_like(sample_logist)
            ],
            axis=1)

        sampled_loss = F.softmax_with_cross_entropy(
            logits=out_logist, label=out_label, soft_label=True)
        return sampled_loss, out_logist, out_label


class Mind_Capsual_Layer(nn.Layer):
    """Mind_Capsual_Layer
    """

    def __init__(self,
                 input_units,
                 output_units,
                 iters=3,
                 maxlen=32,
                 k_max=3,
                 init_std=1.0,
                 batch_size=None):
        super(Mind_Capsual_Layer, self).__init__()

        self.iters = iters
        self.input_units = input_units
        self.output_units = output_units
        self.maxlen = maxlen
        self.init_std = init_std
        self.k_max = k_max
        self.batch_size = batch_size

        # B2I routing
        self.routing_logits = self.create_parameter(
            shape=[1, self.k_max, self.maxlen],
            attr=paddle.ParamAttr(
                name="routing_logits", trainable=False),
            default_initializer=nn.initializer.Normal(
                mean=0.0, std=self.init_std))

        # bilinear mapping
        self.bilinear_mapping_matrix = self.create_parameter(
            shape=[self.input_units, self.output_units],
            attr=paddle.ParamAttr(
                name="bilinear_mapping_matrix", trainable=True),
            default_initializer=nn.initializer.Normal(
                mean=0.0, std=self.init_std))

    def squash(self, Z):
        """squash
        """
        vec_squared_norm = paddle.sum(paddle.square(Z), axis=-1, keepdim=True)
        scalar_factor = vec_squared_norm / \
            (1 + vec_squared_norm) / paddle.sqrt(vec_squared_norm + 1e-8)
        vec_squashed = scalar_factor * Z
        return vec_squashed

    def sequence_mask(self, lengths, maxlen=None, dtype="bool"):
        """sequence_mask
        """
        batch_size = paddle.shape(lengths)[0]
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = paddle.arange(0, maxlen, 1).unsqueeze(0).expand(
            shape=(batch_size, maxlen)).reshape((batch_size, -1, maxlen))
        lengths = lengths.unsqueeze(-1)
        mask = row_vector < lengths
        return mask.astype(dtype)

    def forward(self, item_his_emb, seq_len):
        """forward

        Args:
            item_his_emb : [B, seqlen, dim]
            seq_len : [B, 1]
        """
        batch_size = item_his_emb.shape[0]
        seq_len_tile = paddle.tile(seq_len, [1, self.k_max])

        mask = self.sequence_mask(seq_len_tile, self.maxlen)
        pad = paddle.ones_like(mask, dtype="float32") * (-2**32 + 1)

        # S*e
        low_capsule_new = paddle.matmul(item_his_emb,
                                        self.bilinear_mapping_matrix)

        low_capsule_new_nograd = paddle.assign(low_capsule_new)
        low_capsule_new_nograd.stop_gradient = True

        B = paddle.tile(self.routing_logits,
                        [paddle.shape(item_his_emb)[0], 1, 1])

        for i in range(self.iters - 1):
            B_mask = paddle.where(mask, B, pad)
            # print(B_mask)
            W = F.softmax(B_mask, axis=-1)
            high_capsule_tmp = paddle.matmul(W, low_capsule_new_nograd)
            high_capsule = self.squash(high_capsule_tmp)
            B_delta = paddle.matmul(
                high_capsule, low_capsule_new_nograd, transpose_y=True)
            B += B_delta / paddle.maximum(
                paddle.norm(
                    B_delta, p=2, axis=-1, keepdim=True),
                paddle.ones_like(B_delta))

        B_mask = paddle.where(mask, B, pad)
        W = F.softmax(B_mask, axis=-1)
        # paddle.static.Print(W)
        high_capsule_tmp = paddle.matmul(W, low_capsule_new)
        # high_capsule_tmp.stop_gradient = False

        high_capsule = self.squash(high_capsule_tmp)
        # high_capsule.stop_gradient = False

        return high_capsule, W, seq_len


class MindLayer(nn.Layer):
    """MindLayer
    """

    def __init__(self,
                 item_count,
                 embedding_dim,
                 hidden_size,
                 neg_samples=100,
                 maxlen=30,
                 pow_p=1.0,
                 capsual_iters=3,
                 capsual_max_k=3,
                 capsual_init_std=1.0,
                 batch_size=None):
        super(MindLayer, self).__init__()
        self.pow_p = pow_p
        self.hidden_size = hidden_size
        self.item_count = item_count
        self.item_id_range = paddle.arange(end=item_count, dtype="int64")
        self.item_emb = nn.Embedding(
            item_count,
            embedding_dim,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                name="item_emb",
                initializer=nn.initializer.XavierUniform(
                    fan_in=item_count, fan_out=embedding_dim)))
        self.embedding_bias = self.create_parameter(
            shape=(item_count, ),
            is_bias=True,
            attr=paddle.ParamAttr(
                name="embedding_bias", trainable=False),
            default_initializer=nn.initializer.Constant(0))

        self.capsual_layer = Mind_Capsual_Layer(
            embedding_dim,
            hidden_size,
            maxlen=maxlen,
            iters=capsual_iters,
            k_max=capsual_max_k,
            init_std=capsual_init_std,
            batch_size=batch_size)
        self.sampled_softmax = Mind_SampledSoftmaxLoss_Layer(
            item_count, neg_samples, batch_size=batch_size)

    def label_aware_attention(self, keys, query):
        """label_aware_attention
        """
        weight = paddle.sum(keys * query, axis=-1, keepdim=True)
        weight = paddle.pow(weight, self.pow_p)  # [x,k_max,1]
        weight = F.softmax(weight, axis=1)
        output = paddle.sum(keys * weight, axis=1)
        return output, weight

    def forward(self, hist_item, seqlen, labels=None):
        """forward

        Args:
            hist_item : [B, maxlen, 1]
            seqlen : [B, 1]
            target : [B, 1]
        """

        hit_item_emb = self.item_emb(hist_item)  # [B, seqlen, embed_dim]
        user_cap, cap_weights, cap_mask = self.capsual_layer(hit_item_emb,
                                                             seqlen)
        if not self.training:
            return user_cap, cap_weights
        target_emb = self.item_emb(labels)
        user_emb, W = self.label_aware_attention(user_cap, target_emb)

        return self.sampled_softmax(
            user_emb, labels, self.item_emb.weight,
            self.embedding_bias), W, user_cap, cap_weights, cap_mask
