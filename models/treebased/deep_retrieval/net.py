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


class DeepRetrieval(nn.Layer):
    def __init__(self, width, height, beam_search_num, item_path_volume, user_embedding_size):
        super(DeepRetrieval, self).__init__()
        self.width = width
        self.height = height
        self.beam_search_num = beam_search_num
        self.item_path_volume = item_path_volume
        self.user_embedding_size = user_embedding_size

        in_sizes = [user_embedding_size + i *
                    user_embedding_size for i in range(0, self.width)]
        print("in_sizes: {}".format(in_sizes))
        out_sizes = [self.height] * self.width
        print("out_sizes: {}".format(out_sizes))

        self.mlp_layers = []

        for i in range(width):
            linear = paddle.nn.Linear(
                in_features=in_sizes[i],
                out_features=out_sizes[i],
                weight_attr=paddle.ParamAttr(
                    name="C_{}_mlp_weight".format(i),
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(out_sizes[i]))))
            self.mlp_layers.append(linear)

        self.path_embedding = []
        for i in range(width):
            emb = paddle.nn.Embedding(
                self.height,
                self.user_embedding_size,
                sparse=True,
                weight_attr=paddle.ParamAttr(
                    name="C_{}_path_embedding".format(i),
                    initializer=paddle.nn.initializer.Uniform())
            )
            self.path_embedding.append(emb)

    def forward(self, user_embedding, item_path_kd_label=None, is_infer=False):

        def train_forward():
            # item_path_kd_label: list [ list[ (1, D), ..., (1, D) ],..., ]
            kd_label_list = []
            for idx, all_kd_val in enumerate(item_path_kd_label):
                kd_label_list.append(paddle.concat(
                    all_kd_val, axis=0))  # (J, D)
            kd_label = paddle.concat(
                kd_label_list, axis=0)  # (batch_size * J, D)

            # find path emb idx for every item
            path_emb_idx_lists = []
            for idx in range(self.width):
                cur_path_emb_idx = paddle.slice(
                    kd_label, axes=[1], starts=[idx], ends=[idx+1])  # (batch_size * J, 1)
                path_emb_idx_lists.append(cur_path_emb_idx)

            # Lookup table path emb
            # The main purpose of two-step table lookup is for distributed PS training
            path_emb = []
            for idx in range(self.width):
                emb = self.path_embedding[idx](
                    path_emb_idx_lists[idx])  # (batch_size * J, emb_shape)
                path_emb.append(emb)

            # expand user_embedding (batch_size, emb_shape) -> (batch_size * J, emb_shape)
            input_embedding = paddle.unsqueeze(
                user_embedding, axis=0)
            input_embedding = paddle.expand(input_embedding, shape=[self.item_path_volume,
                                                                    input_embedding.shape[1], input_embedding.shape[2]])
            input_embedding = paddle.reshape(
                input_embedding, (self.item_path_volume * input_embedding.shape[1], input_embedding.shape[2]))

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

            return path_prob

        def infer_forward():
            # beamsearch
            beam_search_num = paddle.full(
                shape=[1, 1], fill_value=self.beam_search_num, dtype='int64')
            height = paddle.full(
                shape=[1, 1], fill_value=self.height, dtype='int64')

            cur_index = []
            prev_index = []
            prev_embedding = []
            prev_index.append([])

            for i in range(self.width):
                if i == 0:
                    # first layer, input only use user embedding
                    tmp_output = F.softmax(self.mlp_layers[i](user_embedding))
                    # assert beam_search_num < height
                    _, index = paddle.topk(
                        tmp_output, self.beam_search_num)

                    # expand user_embedding (batch_size, emb_shape) -> (batch_size * B, emb_shape)
                    input_embedding = paddle.unsqueeze(
                        user_embedding, axis=0)
                    input_embedding = paddle.expand(input_embedding, shape=[self.beam_search_num,
                                                                            input_embedding.shape[1], input_embedding.shape[2]])
                    input_embedding = paddle.reshape(
                        input_embedding, (self.beam_search_num * input_embedding.shape[1], input_embedding.shape[2]))

                    # append user embedding
                    prev_embedding.append(input_embedding)

                    # append cur path embedding
                    input_emb = self.path_embedding[i](index)
                    prev_embedding.append(input_emb)

                    cur_index.append(index)
                else:
                    # other layer, use user embedding + path_embedding

                    # (batch_size, B, 1)
                    prev_prob = paddle.unsqueeze(layer_porb[-1], axis=2)
                    # (batch_size, 1, K)
                    cur_prob = paddle.unsqueeze(tmp_output, axis=1)
                    # (batch_size, B * K)
                    beam_prob = paddle.reshape(paddle.matmul(
                        prev_prob, cur_prob), [-1, self.beam_search_num * self.height])

                    # (batch_size, B)
                    print("beam_prob: {}".format(beam_prob))
                    cur_top_prob, cur_top_index = paddle.topk(
                        beam_prob, self.beam_search_num)
                    layer_porb.append(cur_top_prob)

                    # prev_index of B
                    prev_top_index = paddle.floor_divide(
                        cur_top_index, height)
                    print("cur_top_index: {}, prev_top_index: {}".format(
                        cur_top_index, prev_top_index))

                    # cur_index of K
                    cur_top_abs_index = paddle.mod(
                        cur_top_index, beam_search_num)

                    prev_index.append(prev_top_index)
                    cur_index.append(cur_top_abs_index)

                layer_prob_output.append(tmp_output)
                if i != self.width - 1:
                    tmp_input = paddle.concat([tmp_input, tmp_output], axis=-1)

            kd_path = []
            print("cur_index: {}".format(cur_index))
            print("prev_index: {}".format(prev_index))

            kd_path.append(cur_index[-1])
            print("cur_index[-1]: {}".format(cur_index[-1]))
            for i in range(self.width - 1, 0, -1):
                print("cur_index[{}]: {}".format(i-1, cur_index[i-1]))
                print("prev_index[{}]: {}".format(i, prev_index[i]))
                kd_path.append(paddle.index_sample(
                    cur_index[i-1], prev_index[i]))
            kd_path.reverse()
            print("kd_path: {}".format(kd_path))

            # (batch_size, B, D)
            kd_path_concat = paddle.reshape(paddle.concat(
                kd_path, axis=-1), [-1, self.beam_search_num, self.width])

            print("kd_path_concat: {}".format(kd_path_concat))
            print("layer_porb: {}".format(layer_porb))

            return kd_path_concat, layer_porb[-1]

        if is_infer:
            return infer_forward()
        else:
            return train_forward()
