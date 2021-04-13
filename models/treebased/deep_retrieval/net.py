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

        in_sizes = [user_embedding_size] + [user_embedding_size + i *
                                            self.height for i in range(1, self.width)]
        print("in_sizes: {}".format(in_sizes))
        out_sizes = [self.height] * self.width
        print("out_sizes: {}".format(out_sizes))

        self.mlp_layers = []

        for i in range(width):
            linear = paddle.nn.Linear(
                in_features=in_sizes[i],
                out_features=out_sizes[i],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(out_sizes[i]))))
            self.mlp_layers.append(linear)

    def forward(self, user_embedding, is_infer=False):

        layer_prob_output = []
        tmp_input = user_embedding
        tmp_output = None

        if not is_infer:
            for i in range(self.width):
                tmp_output = F.softmax(self.mlp_layers[i](tmp_input))
                layer_prob_output.append(tmp_output)
                if i != self.width - 1:
                    tmp_input = paddle.concat([tmp_input, tmp_output], axis=-1)
        else:
            # beamsearch
            beam_search_num = paddle.full(
                shape=[1, 1], fill_value=self.beam_search_num, dtype='int64')
            height = paddle.full(
                shape=[1, 1], fill_value=self.height, dtype='int64')

            layer_porb = []
            prev_index = []
            prev_index.append([])
            cur_index = []

            for i in range(self.width):
                tmp_output = F.softmax(self.mlp_layers[i](tmp_input))
                if i == 0:
                    prob, index = paddle.topk(
                        tmp_output, self.beam_search_num)
                    layer_porb.append(prob)
                    cur_index.append(index)
                else:
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

        return layer_prob_output
