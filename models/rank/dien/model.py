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

import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase


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
        hist_item_seq = fluid.data(
            name="hist_item_seq", shape=[None, 1], dtype="int64", lod_level=1)
        self.data_var.append(hist_item_seq)

        hist_cat_seq = fluid.data(
            name="hist_cat_seq", shape=[None, 1], dtype="int64", lod_level=1)
        self.data_var.append(hist_cat_seq)

        target_item = fluid.data(
            name="target_item", shape=[None], dtype="int64")
        self.data_var.append(target_item)

        target_cat = fluid.data(name="target_cat", shape=[None], dtype="int64")
        self.data_var.append(target_cat)

        label = fluid.data(name="label", shape=[None, 1], dtype="float32")
        self.data_var.append(label)

        mask = fluid.data(
            name="mask", shape=[None, seq_len, 1], dtype="float32")
        self.data_var.append(mask)

        target_item_seq = fluid.data(
            name="target_item_seq", shape=[None, seq_len], dtype="int64")
        self.data_var.append(target_item_seq)

        target_cat_seq = fluid.data(
            name="target_cat_seq", shape=[None, seq_len], dtype="int64")
        self.data_var.append(target_cat_seq)

        neg_hist_item_seq = fluid.data(
            name="neg_hist_item_seq",
            shape=[None, 1],
            dtype="int64",
            lod_level=1)
        self.data_var.append(neg_hist_item_seq)

        neg_hist_cat_seq = fluid.data(
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

        concat = fluid.layers.concat(
            [hist, target_expand, hist - target_expand, hist * target_expand],
            axis=2)
        atten_fc1 = fluid.layers.fc(name="atten_fc1",
                                    input=concat,
                                    size=80,
                                    act=self.act,
                                    num_flatten_dims=2)
        atten_fc2 = fluid.layers.fc(name="atten_fc2",
                                    input=atten_fc1,
                                    size=40,
                                    act=self.act,
                                    num_flatten_dims=2)
        atten_fc3 = fluid.layers.fc(name="atten_fc3",
                                    input=atten_fc2,
                                    size=1,
                                    num_flatten_dims=2)
        atten_fc3 += mask
        atten_fc3 = fluid.layers.transpose(x=atten_fc3, perm=[0, 2, 1])
        atten_fc3 = fluid.layers.scale(x=atten_fc3, scale=hidden_size**-0.5)
        weight = fluid.layers.softmax(atten_fc3)
        weighted = fluid.layers.transpose(x=weight, perm=[0, 2, 1])
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

        item_emb_attr = fluid.ParamAttr(name="item_emb")
        cat_emb_attr = fluid.ParamAttr(name="cat_emb")

        # ------------------------- Embedding Layer --------------------------

        hist_item_emb = fluid.embedding(
            input=hist_item_seq,
            size=[self.item_count, self.item_emb_size],
            param_attr=item_emb_attr,
            is_sparse=self.is_sparse)

        neg_hist_cat_emb = fluid.embedding(
            input=neg_hist_cat_seq,
            size=[self.cat_count, self.cat_emb_size],
            param_attr=cat_emb_attr,
            is_sparse=self.is_sparse)

        neg_hist_item_emb = fluid.embedding(
            input=neg_hist_item_seq,
            size=[self.item_count, self.item_emb_size],
            param_attr=item_emb_attr,
            is_sparse=self.is_sparse)

        hist_cat_emb = fluid.embedding(
            input=hist_cat_seq,
            size=[self.cat_count, self.cat_emb_size],
            param_attr=cat_emb_attr,
            is_sparse=self.is_sparse)

        target_item_emb = fluid.embedding(
            input=target_item,
            size=[self.item_count, self.item_emb_size],
            param_attr=item_emb_attr,
            is_sparse=self.is_sparse)

        target_cat_emb = fluid.embedding(
            input=target_cat,
            size=[self.cat_count, self.cat_emb_size],
            param_attr=cat_emb_attr,
            is_sparse=self.is_sparse)

        target_item_seq_emb = fluid.embedding(
            input=target_item_seq,
            size=[self.item_count, self.item_emb_size],
            param_attr=item_emb_attr,
            is_sparse=self.is_sparse)

        target_cat_seq_emb = fluid.embedding(
            input=target_cat_seq,
            size=[self.cat_count, self.cat_emb_size],
            param_attr=cat_emb_attr,
            is_sparse=self.is_sparse)

        item_b = fluid.embedding(
            input=target_item,
            size=[self.item_count, 1],
            param_attr=fluid.initializer.Constant(value=0.0))

        # ------------------------- Interest Extractor Layer --------------------------

        hist_seq_concat = fluid.layers.concat(
            [hist_item_emb, hist_cat_emb], axis=2)
        neg_hist_seq_concat = fluid.layers.concat(
            [neg_hist_item_emb, neg_hist_cat_emb], axis=2)
        target_seq_concat = fluid.layers.concat(
            [target_item_seq_emb, target_cat_seq_emb], axis=2)
        target_concat = fluid.layers.concat(
            [target_item_emb, target_cat_emb], axis=1)

        reshape_hist_item_emb = fluid.layers.reduce_sum(hist_seq_concat, dim=1)
        neg_reshape_hist_item_emb = fluid.layers.reduce_sum(
            neg_hist_seq_concat, dim=1)
        gru_input_hist_item_emb = fluid.layers.concat(
            [reshape_hist_item_emb] * 3, axis=1)

        gru_h1 = fluid.layers.dynamic_gru(
            gru_input_hist_item_emb, size=self.item_emb_size * 2)
        gru_h1_input = fluid.layers.concat([gru_h1] * 3, axis=1)
        gru_h2 = fluid.layers.dynamic_gru(
            gru_h1_input, size=self.item_emb_size * 2)

        # ------------------------- Auxiliary loss  --------------------------

        pad_value = fluid.layers.zeros(shape=[1], dtype='float32')
        start_value = fluid.layers.zeros(shape=[1], dtype='int32')
        gru_out_pad, lengths = fluid.layers.sequence_pad(gru_h2, pad_value)
        pos_seq_pad, _ = fluid.layers.sequence_pad(reshape_hist_item_emb,
                                                   pad_value)
        neg_seq_pad, _ = fluid.layers.sequence_pad(neg_reshape_hist_item_emb,
                                                   pad_value)
        seq_shape = fluid.layers.shape(pos_seq_pad)
        
        if(seq_shape[1] < 2):
            aux_loss = 0
        else:
            test_pos = fluid.layers.reduce_sum(
                fluid.layers.reduce_sum(
                    fluid.layers.log(
                        fluid.layers.sigmoid(
                            fluid.layers.reduce_sum(
                                gru_out_pad[:, start_value:seq_shape[1] - 1, :]
                                * pos_seq_pad[:, start_value + 1:seq_shape[
                                    1], :],
                                dim=2,
                                keep_dim=True))),
                    dim=2),
                dim=1,
                keep_dim=True)
            
            test_neg = fluid.layers.reduce_sum(
                fluid.layers.log(
                    fluid.layers.sigmoid(
                        fluid.layers.reduce_sum(
                            gru_out_pad[:, start_value:seq_shape[1] - 1, :]
                            * neg_seq_pad[:, start_value + 1:seq_shape[
                                1], :],
                            dim=2,
                            keep_dim=True))),
                dim=2),
            dim=1,
            keep_dim=True)
            
            aux_loss = fluid.layers.mean(test_neg + test_pos)

        # ------------------------- Interest Evolving Layer (GRU with attentional input (AIGRU)) --------------------------

        weighted_vector = self.din_attention(gru_out_pad, target_seq_concat,
                                             mask)
        weighted_vector = fluid.layers.transpose(weighted_vector, [1, 0, 2])
        concat_weighted_vector = fluid.layers.concat(
            [weighted_vector] * 3, axis=2)

        attention_rnn = fluid.layers.StaticRNN(name="attnention_evolution")

        with attention_rnn.step():
            word = attention_rnn.step_input(concat_weighted_vector)
            prev = attention_rnn.memory(
                shape=[-1, self.item_emb_size * 2], batch_ref=word)
            hidden, _, _ = fluid.layers.gru_unit(
                input=word, hidden=prev, size=self.item_emb_size * 6)
            attention_rnn.update_memory(prev, hidden)
            attention_rnn.output(hidden)

        attention_rnn_res = attention_rnn()
        attention_rnn_res_T = fluid.layers.transpose(attention_rnn_res,
                                                     [1, 0, 2])[:, -1, :]

        out = fluid.layers.sequence_pool(input=hist_item_emb, pool_type='sum')
        out_fc = fluid.layers.fc(name="out_fc",
                                 input=out,
                                 size=self.item_emb_size + self.cat_emb_size,
                                 num_flatten_dims=1)
        embedding_concat = fluid.layers.concat(
            [attention_rnn_res_T, target_concat], axis=1)

        fc1 = fluid.layers.fc(name="fc1",
                              input=embedding_concat,
                              size=80,
                              act=self.act)
        fc2 = fluid.layers.fc(name="fc2", input=fc1, size=40, act=self.act)
        fc3 = fluid.layers.fc(name="fc3", input=fc2, size=1)
        logit = fc3 + item_b

        loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=logit, label=label)

        avg_loss = fluid.layers.mean(loss) + aux_loss
        self._cost = avg_loss

        self.predict = fluid.layers.sigmoid(logit)
        predict_2d = fluid.layers.concat([1 - self.predict, self.predict], 1)
        label_int = fluid.layers.cast(label, 'int64')
        auc_var, batch_auc_var, _ = fluid.layers.auc(input=predict_2d,
                                                     label=label_int,
                                                     slide_steps=0)
        self._metrics["AUC"] = auc_var
        self._metrics["BATCH_AUC"] = batch_auc_var
        if is_infer:
            self._infer_results["AUC"] = auc_var
