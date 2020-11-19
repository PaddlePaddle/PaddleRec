#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase

import numpy as np
import paddle


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        self.item_emb_size = envs.get_global_env(
            "hyper_parameters.item_emb_size", 64)
        self.cat_emb_size = envs.get_global_env(
            "hyper_parameters.cat_emb_size", 64)
        self.act = envs.get_global_env("hyper_parameters.act", "sigmoid")
        self.is_sparse = envs.get_global_env("hyper_parameters.is_sparse",
                                             False)
        # significant for speeding up the training process
        self.use_DataLoader = envs.get_global_env(
            "hyper_parameters.use_DataLoader", False)
        self.item_count = envs.get_global_env("hyper_parameters.item_count",
                                              63001)
        self.cat_count = envs.get_global_env("hyper_parameters.cat_count", 801)

    def input_data(self, is_infer=False, **kwargs):
        seq_len = -1
        self.data_var = []
        hist_item_seq = paddle.static.data(
            name="hist_item_seq", shape=[None, 1], dtype="int64", lod_level=1)
        self.data_var.append(hist_item_seq)

        hist_cat_seq = paddle.static.data(
            name="hist_cat_seq", shape=[None, 1], dtype="int64", lod_level=1)
        self.data_var.append(hist_cat_seq)

        target_item = paddle.static.data(
            name="target_item", shape=[None], dtype="int64")
        self.data_var.append(target_item)

        target_cat = paddle.static.data(
            name="target_cat", shape=[None], dtype="int64")
        self.data_var.append(target_cat)

        label = paddle.static.data(
            name="label", shape=[None, 1], dtype="float32")
        self.data_var.append(label)

        mask = paddle.static.data(
            name="mask", shape=[None, seq_len, 1], dtype="float32")
        self.data_var.append(mask)

        target_item_seq = paddle.static.data(
            name="target_item_seq", shape=[None, seq_len], dtype="int64")
        self.data_var.append(target_item_seq)

        target_cat_seq = paddle.static.data(
            name="target_cat_seq", shape=[None, seq_len], dtype="int64")
        self.data_var.append(target_cat_seq)

        neg_hist_item_seq = paddle.static.data(
            name="neg_hist_item_seq",
            shape=[None, 1],
            dtype="int64",
            lod_level=1)
        self.data_var.append(neg_hist_item_seq)

        neg_hist_cat_seq = paddle.static.data(
            name="neg_hist_cat_seq",
            shape=[None, 1],
            dtype="int64",
            lod_level=1)
        self.data_var.append(neg_hist_cat_seq)

        train_inputs = [hist_item_seq] + [hist_cat_seq] + [target_item] + [
            target_cat
        ] + [label] + [mask] + [target_item_seq] + [target_cat_seq] + [
            neg_hist_item_seq
        ] + [neg_hist_cat_seq]
        return train_inputs

    def din_attention(self, hist, target_expand, mask, return_alpha=False):
        """activation weight"""

        hidden_size = hist.shape[-1]

        concat = paddle.concat(
            x=[
                hist, target_expand, hist - target_expand, hist * target_expand
            ],
            axis=2)
        atten_fc1 = paddle.static.nn.fc(name="atten_fc1",
                                        x=concat,
                                        size=80,
                                        activation=self.act,
                                        num_flatten_dims=2)
        atten_fc2 = paddle.static.nn.fc(name="atten_fc2",
                                        x=atten_fc1,
                                        size=40,
                                        activation=self.act,
                                        num_flatten_dims=2)
        atten_fc3 = paddle.static.nn.fc(name="atten_fc3",
                                        x=atten_fc2,
                                        size=1,
                                        num_flatten_dims=2)
        atten_fc3 += mask
        atten_fc3 = paddle.transpose(x=atten_fc3, perm=[0, 2, 1])
        atten_fc3 = paddle.scale(x=atten_fc3, scale=hidden_size**-0.5)
        weight = paddle.nn.functional.softmax(x=atten_fc3)
        weighted = paddle.transpose(x=weight, perm=[0, 2, 1])
        weighted_vector = weighted * hist
        if return_alpha:
            return hist, weighted
        return weighted_vector

    def net(self, inputs, is_infer=False):

        # ------------------------- network input --------------------------

        hist_item_seq = inputs[0]  # history item sequence
        hist_cat_seq = inputs[1]  # history category sequence
        target_item = inputs[2]  # one dim target item
        target_cat = inputs[3]  # one dim target category
        label = inputs[4]  # label
        mask = inputs[5]  # mask
        target_item_seq = inputs[6]  # target item expand to sequence
        target_cat_seq = inputs[7]  # traget category expand to sequence
        neg_hist_item_seq = inputs[8]  # neg item sampling for aux loss
        neg_hist_cat_seq = inputs[9]  # neg cat sampling for aux loss

        item_emb_attr = paddle.ParamAttr(name="item_emb")
        cur_program = paddle.static.Program()
        cur_block = cur_program.current_block()
        item_emb_copy = cur_block.create_var(
            name="item_emb",
            shape=[self.item_count, self.item_emb_size],
            dtype='float32')
        #item_emb_copy = fluid.layers.Print(item_emb_copy, message="Testing:")
        ##item_emb_attr = fluid.layers.Print(item_emb_attr, summarize=2)
        cat_emb_attr = paddle.ParamAttr(name="cat_emb")

        # ------------------------- Embedding Layer --------------------------

        hist_item_emb = paddle.static.nn.embedding(
            input=hist_item_seq,
            size=[self.item_count, self.item_emb_size],
            param_attr=item_emb_attr,
            is_sparse=self.is_sparse)
        item_emb_copy = paddle.static.Print(
            input=item_emb_copy,
            message="Testing:",
            summarize=20,
            print_phase='backward')
        neg_hist_cat_emb = paddle.static.nn.embedding(
            input=neg_hist_cat_seq,
            size=[self.cat_count, self.cat_emb_size],
            param_attr=cat_emb_attr,
            is_sparse=self.is_sparse)

        neg_hist_item_emb = paddle.static.nn.embedding(
            input=neg_hist_item_seq,
            size=[self.item_count, self.item_emb_size],
            param_attr=item_emb_attr,
            is_sparse=self.is_sparse)

        hist_cat_emb = paddle.static.nn.embedding(
            input=hist_cat_seq,
            size=[self.cat_count, self.cat_emb_size],
            param_attr=cat_emb_attr,
            is_sparse=self.is_sparse)

        target_item_emb = paddle.static.nn.embedding(
            input=target_item,
            size=[self.item_count, self.item_emb_size],
            param_attr=item_emb_attr,
            is_sparse=self.is_sparse)

        target_cat_emb = paddle.static.nn.embedding(
            input=target_cat,
            size=[self.cat_count, self.cat_emb_size],
            param_attr=cat_emb_attr,
            is_sparse=self.is_sparse)

        target_item_seq_emb = paddle.static.nn.embedding(
            input=target_item_seq,
            size=[self.item_count, self.item_emb_size],
            param_attr=item_emb_attr,
            is_sparse=self.is_sparse)

        target_cat_seq_emb = paddle.static.nn.embedding(
            input=target_cat_seq,
            size=[self.cat_count, self.cat_emb_size],
            param_attr=cat_emb_attr,
            is_sparse=self.is_sparse)

        item_b = paddle.static.nn.embedding(
            input=target_item,
            size=[self.item_count, 1],
            param_attr=paddle.nn.initializer.Constant(value=0.0))

        # ------------------------- Interest Extractor Layer --------------------------

        hist_seq_concat = paddle.concat(
            x=[hist_item_emb, hist_cat_emb], axis=2)
        neg_hist_seq_concat = paddle.concat(
            x=[neg_hist_item_emb, neg_hist_cat_emb], axis=2)
        target_seq_concat = paddle.concat(
            x=[target_item_seq_emb, target_cat_seq_emb], axis=2)
        target_concat = paddle.concat(
            x=[target_item_emb, target_cat_emb], axis=1)

        reshape_hist_item_emb = paddle.sum(x=hist_seq_concat, axis=1)
        neg_reshape_hist_item_emb = paddle.sum(x=neg_hist_seq_concat, axis=1)
        gru_input_hist_item_emb = paddle.concat(
            x=[reshape_hist_item_emb] * 3, axis=1)

        gru_h1 = paddle.fluid.layers.dynamic_gru(
            gru_input_hist_item_emb, size=self.item_emb_size * 2)
        gru_h1_input = paddle.concat(x=[gru_h1] * 3, axis=1)
        gru_h2 = paddle.fluid.layers.dynamic_gru(
            gru_h1_input, size=self.item_emb_size * 2)

        # ------------------------- Auxiliary loss  --------------------------

        pad_value = paddle.zeros(shape=[1], dtype='float32')
        start_value = paddle.zeros(shape=[1], dtype='int32')
        gru_out_pad, lengths = paddle.fluid.layers.sequence_pad(gru_h2,
                                                                pad_value)
        pos_seq_pad, _ = paddle.fluid.layers.sequence_pad(
            reshape_hist_item_emb, pad_value)
        neg_seq_pad, _ = paddle.fluid.layers.sequence_pad(
            neg_reshape_hist_item_emb, pad_value)
        seq_shape = paddle.shape(pos_seq_pad)
        if (seq_shape[1] == 1):
            aux_loss = 0
        else:
            test_pos = paddle.sum(x=paddle.sum(x=paddle.log(
                paddle.nn.functional.sigmoid(
                    paddle.sum(x=gru_out_pad[:, start_value:seq_shape[
                        1] - 1, :] * pos_seq_pad[:, start_value + 1:seq_shape[
                            1], :],
                               axis=2,
                               keepdim=True))),
                                               axis=2),
                                  axis=1,
                                  keepdim=True)
            test_neg = paddle.sum(x=paddle.sum(x=paddle.log(
                paddle.nn.functional.sigmoid(
                    paddle.sum(x=gru_out_pad[:, start_value:seq_shape[
                        1] - 1, :] * neg_seq_pad[:, start_value + 1:seq_shape[
                            1], :],
                               axis=2,
                               keepdim=True))),
                                               axis=2),
                                  axis=1,
                                  keepdim=True)
            aux_loss = paddle.mean(x=test_neg + test_pos)

        # ------------------------- Interest Evolving Layer (GRU with attentional input (AIGRU)) --------------------------

        weighted_vector = self.din_attention(gru_out_pad, target_seq_concat,
                                             mask)
        weighted_vector = paddle.transpose(weighted_vector, [1, 0, 2])
        concat_weighted_vector = paddle.concat(x=[weighted_vector] * 3, axis=2)

        attention_rnn = paddle.fluid.layers.StaticRNN(
            name="attention_evolution")

        with attention_rnn.step():
            word = attention_rnn.step_input(concat_weighted_vector)
            prev = attention_rnn.memory(
                shape=[-1, self.item_emb_size * 2], batch_ref=word)
            hidden, _, _ = paddle.fluid.layers.gru_unit(
                input=word, hidden=prev, size=self.item_emb_size * 6)
            attention_rnn.update_memory(prev, hidden)
            attention_rnn.output(hidden)

        attention_rnn_res = attention_rnn()
        attention_rnn_res_T = paddle.transpose(attention_rnn_res,
                                               [1, 0, 2])[:, -1, :]

        out = paddle.fluid.layers.sequence_pool(
            input=hist_item_emb, pool_type='sum')
        out_fc = paddle.static.nn.fc(
            name="out_fc",
            x=out,
            size=self.item_emb_size + self.cat_emb_size,
            num_flatten_dims=1)
        embedding_concat = paddle.concat(
            x=[attention_rnn_res_T, target_concat], axis=1)

        fc1 = paddle.static.nn.fc(name="fc1",
                                  x=embedding_concat,
                                  size=80,
                                  activation=self.act)
        fc2 = paddle.static.nn.fc(name="fc2",
                                  x=fc1,
                                  size=40,
                                  activation=self.act)
        fc3 = paddle.static.nn.fc(name="fc3", x=fc2, size=1)
        logit = fc3 + item_b

        loss = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(
            x=logit, label=label)

        avg_loss = paddle.mean(x=loss) + aux_loss
        self._cost = avg_loss

        self.predict = paddle.nn.functional.sigmoid(logit)
        predict_2d = paddle.concat(x=[1 - self.predict, self.predict], axis=1)

        label_int = paddle.cast(label, 'int64')
        auc_var, batch_auc_var, _ = paddle.fluid.layers.auc(input=predict_2d,
                                                            label=label_int,
                                                            slide_steps=0)
        self._metrics["AUC"] = auc_var
        self._metrics["BATCH_AUC"] = batch_auc_var

        if is_infer:
            self._infer_results["AUC"] = auc_var
