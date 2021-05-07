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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import os
import paddle.fluid as fluid


class DeepRetrieval(nn.Layer):

    def expand_layer(self,input, n):
        # expand input (batch_size, shape) -> (batch_size * n, shape)
        col_size = input.shape[1]
        arr = [input] * n

        input = paddle.concat(arr, axis=1)
        input = paddle.reshape(input, [-1, col_size])
        return input

    def __init__(self, width, height, beam_search_num, item_path_volume, user_embedding_size, item_count,
                 use_multi_task_learning=False, multi_task_mlp_size=None, is_static=False):
        super(DeepRetrieval, self).__init__()
        self.width = width
        self.height = height
        self.beam_search_num = beam_search_num
        self.item_path_volume = item_path_volume
        self.user_embedding_size = user_embedding_size

        in_sizes = [user_embedding_size + i *
                    user_embedding_size for i in range(self.width)]
        print("in_sizes: {}".format(in_sizes))
        out_sizes = [self.height] * self.width
        print("out_sizes: {}".format(out_sizes))
        self.use_multi_task_learning = use_multi_task_learning
        self.mlp_layers = []
        self.multi_task_mlp_layers_size = [user_embedding_size]
        self.multi_task_mlp_layers = []
        for i in range(width):
            linear = paddle.nn.Linear(
                in_features=in_sizes[i],
                out_features=out_sizes[i],
                weight_attr=paddle.ParamAttr(
                    name="C_{}_mlp_weight".format(i),
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(out_sizes[i]))))
            self.mlp_layers.append(linear)
            self.add_sublayer("C_{}_mlp_weight".format(i), linear)

        self.path_embedding = paddle.nn.Embedding(
            self.height,
            self.user_embedding_size,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="path_embedding",
                initializer=paddle.nn.initializer.Uniform())
        )
        if is_static:
            self.em_startup_program = fluid.default_startup_program().clone()
            self.em_main_program = paddle.static.default_main_program().clone()
        if self.use_multi_task_learning:
            self.item_count = item_count
            for i in multi_task_mlp_size:
                self.multi_task_mlp_layers_size.append(i)
            for i in range(len(self.multi_task_mlp_layers_size) - 1):
                linear = paddle.nn.Linear(
                    in_features=self.multi_task_mlp_layers_size[i],
                    out_features=self.multi_task_mlp_layers_size[i + 1],
                    weight_attr=paddle.ParamAttr(
                        name="multi_task_{}_mlp_weight".format(i),
                        initializer=paddle.nn.initializer.Normal(
                            std=1.0 / math.sqrt(out_sizes[i]))))
                self.multi_task_mlp_layers.append(linear)
                self.add_sublayer("multi_task_{}_mlp_weight".format(i), linear)
            self.dot_product_size = self.multi_task_mlp_layers_size[-1]
            self.multi_task_item_embedding = paddle.nn.Embedding(
                self.item_count,
                self.dot_product_size,
                weight_attr=paddle.ParamAttr(
                    name="multi_task_item_embedding_weight",
                    initializer=paddle.nn.initializer.Uniform()))

    def generate_candidate_path_for_item(self, input_embeddings, beam_size):
        if beam_size > self.height:
            beam_size = self.height
        height = paddle.full(
            shape=[1, 1], fill_value=self.height, dtype='int64')
        prob_list = []
        saved_path = None
        w = []
        row = row = paddle.zeros_like(input_embeddings,dtype="int32")
        row = paddle.sum(row, axis=-1)
        row = paddle.reshape(row, [-1, 1])

        for i in range(beam_size):

            x = row + i
            w.append(x)
        # [batch] all zeros
        batch_row = paddle.reshape(row, [-1])
        row = paddle.concat(w, axis=-1)
        # row = [0,1,2...beam-1,0,1,2....] ,shape = [beam *batch]
        row = paddle.reshape(row, [-1])
        for i in range(self.width):
            if i == 0:
                # [batch, height]
                pro = F.softmax(self.mlp_layers[0](input_embeddings))
                # [height]
                pro_sum = paddle.sum(pro, axis=0)
                # [beam_size],[beam_size]
                _, index = paddle.topk(pro_sum, beam_size)
                # [1, beam_size]
                saved_path = paddle.unsqueeze(index, 0)
                # [1,beam_size]
                # multi_index = paddle.unsqueeze(index, 0)
                # multi_index = paddle.expand(multi_index, [batch_size, beam_size])
                # # [batch_size,beam_size]
                # last_prob = paddle.index_sample(pro, multi_index)
                # [batch_size,beam_size]
                last_prob = paddle.index_select(paddle.reshape(pro,[self.height,-1]),index)
                last_prob = paddle.reshape(last_prob, [-1, beam_size])
                prob_list.append(last_prob)
                # [batch * beam, emb_size]
                input_embeddings = self.expand_layer(input_embeddings, beam_size)
                # # [batch, 1, emb_size]
                # input_embeddings = paddle.unsqueeze(input_embeddings, 1)
                # # [batch,beam,emb_size]
                input_embeddings = paddle.reshape(input_embeddings, [-1, beam_size, self.user_embedding_size])
            else:
                # [beam, i]
                reverse_saved_path = paddle.transpose(saved_path, [1, 0])
                # [beam, i,emb_size ]
                saved_path_emb = self.path_embedding(reverse_saved_path)
                # [beam, i * emb_size ]
                input = paddle.reshape(saved_path_emb, [beam_size, -1])

                # [beam * batch, i * emb_size]
                input = paddle.index_select(input, row)
                print("input shape ",input.shape)
                # [batch, beam, i * emb_size]
                input = paddle.reshape(input,[-1, beam_size, i * self.user_embedding_size])
                # # input = paddle.concat(emb_list,axis=-1)
                # input = paddle.unsqueeze(input, 0)
                # # [batch, beam, i * emb_size]
                # input = paddle.expand(input, [batch_size, beam_size, i * self.user_embedding_size])

                # [batch, beam, (i+1) * emb_size]
                input = paddle.concat([input_embeddings, input], axis=-1)
                # [batch, beam_size, height]
                out = F.softmax(self.mlp_layers[i](input))
                # [beam, height]
                pro_sum = paddle.sum(out, axis=0)
                # [beam * height]
                pro_sum = paddle.reshape(pro_sum, [-1])
                # [beam]
                _, index = paddle.topk(pro_sum, beam_size)
                # [1,beam]
                beam_index = paddle.floor_divide(index, height)
                item_index = paddle.mod(index, height)
                # [batch,beam]
                batch_beam_index = paddle.index_select(beam_index, batch_row)
                # batch_beam_index = paddle.expand(beam_index, [batch_size, beam_size])
                # [i,beam_size]
                saved_path_index = paddle.expand(beam_index, [saved_path.shape[0], beam_size])
                saved_path = paddle.index_sample(saved_path, saved_path_index)
                for j in range(len(prob_list)):
                    prob_list[j] = paddle.index_sample(prob_list[j], batch_beam_index)

                # [i + 1,emb_size]
                # [i + 1,emb_size]
                saved_path = paddle.concat([saved_path, item_index], axis=0)

            # [beam, width]
        saved_path = paddle.transpose(saved_path, [1, 0])

        # [batch, beam]
        final_prob = prob_list[0]
        for i in range(1, len(prob_list)):
            final_prob = paddle.multiply(final_prob, prob_list[i])

            # [beam]
        final_prob = paddle.sum(final_prob, axis=0)

        return saved_path, final_prob

    def forward(self, user_embedding, kd_label=None, multi_task_positive_labels=None,
                multi_task_negative_labels=None, is_infer=False):

        print("in train forward")
        print("user_emb", user_embedding)
        print("kd-Label", kd_label)
        print("pos_label", multi_task_positive_labels)

        # print("item_path_kd_label = ", item_path_kd_label)

        def train_forward(user_embedding, kd_label=None, multi_task_positive_labels=None,
                          multi_task_negative_labels=None):
            # item_path_kd_label: list [ list[ (1, D), ..., (1, D) ],..., ]
            # print("in train forward ",user_embedding.shape)
            # kd_label_list = []
            # batch_size = len(item_path_kd_label)
            # print("batch_size = ",batch_size)
            # print(item_path_kd_label)
            # J = self.item_path_volume
            # D = self.width
            # print(item_path_kd_label[0][0].shape)
            # for idx, all_kd_val in enumerate(item_path_kd_label):
            #     kd_label_list.append(paddle.concat(
            #         all_kd_val, axis=0))  # (J, D)
            # kd_label = paddle.concat(
            #     kd_label_list, axis=0)  # (batch_size * J, D)
            # print(kd_label.shape)
            # print("kd shape ",batch_size * J," ",D)

            # find path emb idx for every item
            print("user_emb--$$$$$", user_embedding)
            print("kd-Label_shape---$$$$$$$", kd_label.shape)
            print("multi-pos-label-$$$$$$$$", multi_task_positive_labels)
            print("multi-neg-label--$$$$$$$$$$$$", multi_task_negative_labels)
            kd_label = paddle.reshape(kd_label, [-1, self.width])
            path_emb_idx_lists = []
            for idx in range(self.width):
                cur_path_emb_idx = paddle.slice(
                    kd_label, axes=[1], starts=[idx], ends=[idx + 1])  # (batch_size * J, 1)
                # print(cur_path_emb_idx.shape)
                # print("cur_path_emb_idx shape",batch_size *J," 1")
                path_emb_idx_lists.append(cur_path_emb_idx)

                print("path_emb_idx.shape", cur_path_emb_idx.shape)
            # Lookup table path emb
            # The main purpose of two-step table lookup is for distributed PS training
            path_emb = []
            for idx in range(self.width):
                emb = self.path_embedding(
                    path_emb_idx_lists[idx])  # (batch_size * J, 1, emb_shape)
                path_emb.append(emb)

                print("emb_shape ", emb.shape)

            # expand user_embedding (batch_size, emb_shape) -> (batch_size * J, emb_shape)

            input_embedding = self.expand_layer(
                user_embedding, self.item_path_volume)

            # calc prob of every layer
            path_prob_list = []
            for i in range(self.width):
                cur_input_list = []
                cur_input = None
                # input: user emb + c_d_emb
                if i == 0:
                    cur_input = input_embedding
                else:
                    cur_input_list.append(input_embedding)
                    for j in range(i):
                        cur_input_list.append(paddle.reshape(
                            path_emb[j], (-1, self.user_embedding_size)))
                    cur_input = paddle.concat(cur_input_list, axis=1)

                layer_prob = F.softmax(self.mlp_layers[i](cur_input))
                cur_path_prob = paddle.index_sample(
                    layer_prob, path_emb_idx_lists[i])  # (batch_size * J, 1)
                path_prob_list.append(cur_path_prob)

            path_prob = paddle.concat(
                path_prob_list, axis=1)  # (batch_size * J, D)

            multi_task_loss = None
            if self.use_multi_task_learning:
                temp = user_embedding
                # (batch, dot_product_size)
                for i in range(len(self.multi_task_mlp_layers)):
                    temp = self.multi_task_mlp_layers[i](temp)

                # (batch, dot_product_size)
                pos_item_embedding = self.multi_task_item_embedding(multi_task_positive_labels)
                neg_item_embedding = self.multi_task_item_embedding(multi_task_negative_labels)

                pos_item_embedding = paddle.reshape(pos_item_embedding, [-1, self.dot_product_size])
                neg_item_embedding = paddle.reshape(pos_item_embedding, [-1, self.dot_product_size])
                # (batch,1)
                pos = paddle.dot(temp, pos_item_embedding)
                neg = paddle.dot(temp, neg_item_embedding)
                neg = paddle.clip(x=neg, min=-15, max=15)

                pos = paddle.log(paddle.nn.functional.sigmoid(pos))
                neg = paddle.log(1 - paddle.nn.functional.sigmoid(neg))
                # (batch,2)
                sum = paddle.concat([pos, neg], axis=1)
                multi_task_loss = paddle.sum(sum)[0]
                multi_task_loss = multi_task_loss * -1

            return path_prob, multi_task_loss

        def infer_forward(user_embedding):

            height = paddle.full(
                shape=[1, 1], fill_value=self.height, dtype='int64')

            prev_index = []
            path_prob = []

            for i in range(self.width):
                if i == 0:
                    # first layer, input only use user embedding
                    # user-embedding [batch, emb_shape]
                    # [batch, K]
                    tmp_output = F.softmax(self.mlp_layers[i](user_embedding))
                    # assert beam_search_num < height
                    # [batch, B]
                    prob, index = paddle.topk(
                        tmp_output, self.beam_search_num)
                    path_prob.append(prob)

                    # expand user_embedding (batch_size, emb_shape) -> (batch_size * B, emb_shape)
                    input_embedding = self.expand_layer(
                        user_embedding, self.beam_search_num)
                    prev_index.append(index)
                    # print("fist prev_index: {}".format(prev_index))

                else:
                    # other layer, use user embedding + path_embedding
                    # (batch_size * B, emb_size * N)
                    # cur_layer_input = paddle.concat(prev_embedding, axis=1)
                    input = input_embedding
                    for j in range(len(prev_index)):
                        # [batch,beam,emb_size]
                        emb = self.path_embedding(prev_index[j])
                        # [batch*beam,emb_size]
                        emb = paddle.reshape(emb, [-1, self.user_embedding_size])
                        input = paddle.concat([input, emb], axis=1)

                    # (batch_size * B, K)
                    # tmp_output = F.softmax(self.mlp_layers[i](cur_layer_input))
                    tmp_output = F.softmax(self.mlp_layers[i](input))
                    # (batch_size, B * K)
                    tmp_output = paddle.reshape(
                        tmp_output, (-1, self.beam_search_num * self.height))
                    # (batch_size, B)
                    prob, index = paddle.topk(
                        tmp_output, self.beam_search_num)
                    # path_prob.append(prob)

                    # prev_index of B
                    # print("index: {}".format(index))
                    prev_top_index = paddle.floor_divide(
                        index, height)
                    # print("prev_top_index: {}".format(prev_top_index))
                    for j in range(len(prev_index)):  #
                        prev_index[j] = paddle.index_sample(prev_index[j], prev_top_index)
                        path_prob[j] = paddle.index_sample(path_prob[j], prev_top_index)
                    path_prob.append(prob)
                    cur_top_abs_index = paddle.mod(
                        index, height)
                    prev_index.append(cur_top_abs_index)
                    # print("cur_top_abs_index: {}".format(cur_top_abs_index))

            final_prob = path_prob[0]
            for i in range(1, len(path_prob)):
                final_prob = paddle.multiply(final_prob, path_prob[i])
            for i in range(len(prev_index)):
                # [batch,beam,1]
                prev_index[i] = paddle.reshape(prev_index[i], [-1, self.beam_search_num, 1])

            # [batch,beam,width],[batch,beam]
            kd_path = paddle.concat(prev_index, axis=-1)
            # print("kd_path", kd_path)
            # print("final_prob", final_prob)
            return kd_path, final_prob

        if is_infer:
            return infer_forward(user_embedding)
        else:
            return train_forward(user_embedding, kd_label, multi_task_positive_labels,
                                 multi_task_negative_labels)