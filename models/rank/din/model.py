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
from paddlerec.core.model import Model as ModelBase


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
        #significant for speeding up the training process
        self.config_path = envs.get_global_env("hyper_parameters.config_path",
                                               "data/config.txt")
        self.use_DataLoader = envs.get_global_env(
            "hyper_parameters.use_DataLoader", False)

    def input_data(self, is_infer=False, **kwargs):
        seq_len = -1
        self.data_var = []
        hist_item_seq = fluid.data(
            name="hist_item_seq", shape=[None, seq_len], dtype="int64")
        self.data_var.append(hist_item_seq)

        hist_cat_seq = fluid.data(
            name="hist_cat_seq", shape=[None, seq_len], dtype="int64")
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

        #if self.use_DataLoader:
        #    self._data_loader = fluid.io.DataLoader.from_generator(
        #        feed_list=self.data_var,
        #        capacity=10000,
        #        use_double_buffer=False,
        #        iterable=False)
        train_inputs = [hist_item_seq] + [hist_cat_seq] + [target_item] + [
            target_cat
        ] + [label] + [mask] + [target_item_seq] + [target_cat_seq]
        return train_inputs

    def config_read(self, config_path):
        with open(config_path, "r") as fin:
            user_count = int(fin.readline().strip())
            item_count = int(fin.readline().strip())
            cat_count = int(fin.readline().strip())
        return user_count, item_count, cat_count

    def din_attention(self, hist, target_expand, mask):
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
        out = fluid.layers.matmul(weight, hist)
        out = fluid.layers.reshape(x=out, shape=[0, hidden_size])
        return out

    def net(self, inputs, is_infer=False):
        hist_item_seq = inputs[0]
        hist_cat_seq = inputs[1]
        target_item = inputs[2]
        target_cat = inputs[3]
        label = inputs[4]
        mask = inputs[5]
        target_item_seq = inputs[6]
        target_cat_seq = inputs[7]

        user_count, item_count, cat_count = self.config_read(self.config_path)

        item_emb_attr = fluid.ParamAttr(name="item_emb")
        cat_emb_attr = fluid.ParamAttr(name="cat_emb")

        hist_item_emb = fluid.embedding(
            input=hist_item_seq,
            size=[item_count, self.item_emb_size],
            param_attr=item_emb_attr,
            is_sparse=self.is_sparse)

        hist_cat_emb = fluid.embedding(
            input=hist_cat_seq,
            size=[cat_count, self.cat_emb_size],
            param_attr=cat_emb_attr,
            is_sparse=self.is_sparse)

        target_item_emb = fluid.embedding(
            input=target_item,
            size=[item_count, self.item_emb_size],
            param_attr=item_emb_attr,
            is_sparse=self.is_sparse)

        target_cat_emb = fluid.embedding(
            input=target_cat,
            size=[cat_count, self.cat_emb_size],
            param_attr=cat_emb_attr,
            is_sparse=self.is_sparse)

        target_item_seq_emb = fluid.embedding(
            input=target_item_seq,
            size=[item_count, self.item_emb_size],
            param_attr=item_emb_attr,
            is_sparse=self.is_sparse)

        target_cat_seq_emb = fluid.embedding(
            input=target_cat_seq,
            size=[cat_count, self.cat_emb_size],
            param_attr=cat_emb_attr,
            is_sparse=self.is_sparse)

        item_b = fluid.embedding(
            input=target_item,
            size=[item_count, 1],
            param_attr=fluid.initializer.Constant(value=0.0))

        hist_seq_concat = fluid.layers.concat(
            [hist_item_emb, hist_cat_emb], axis=2)
        target_seq_concat = fluid.layers.concat(
            [target_item_seq_emb, target_cat_seq_emb], axis=2)
        target_concat = fluid.layers.concat(
            [target_item_emb, target_cat_emb], axis=1)

        out = self.din_attention(hist_seq_concat, target_seq_concat, mask)
        out_fc = fluid.layers.fc(name="out_fc",
                                 input=out,
                                 size=self.item_emb_size + self.cat_emb_size,
                                 num_flatten_dims=1)
        embedding_concat = fluid.layers.concat([out_fc, target_concat], axis=1)

        fc1 = fluid.layers.fc(name="fc1",
                              input=embedding_concat,
                              size=80,
                              act=self.act)
        fc2 = fluid.layers.fc(name="fc2", input=fc1, size=40, act=self.act)
        fc3 = fluid.layers.fc(name="fc3", input=fc2, size=1)
        logit = fc3 + item_b

        loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=logit, label=label)

        avg_loss = fluid.layers.mean(loss)
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
