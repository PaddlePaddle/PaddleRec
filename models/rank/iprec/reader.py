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
from __future__ import print_function
import numpy as np
from paddle.io import Dataset
import random
import json
import numpy as np
import paddle


class RecDataset(Dataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.data = [json.loads(x) for x in open(file_list[0])]
        seed = config.get("hyper_parameters.seed")
        np.random.seed(2021)
        self.f_max_len = config.get("hyper_parameters.f_max_len", 20)
        self.u_max_i = config.get("hyper_parameters.u_max_i", 99)
        self.u_max_f = config.get("hyper_parameters.u_max_f", 220)
        self.u_max_pack = config.get("hyper_parameters.u_max_pack", 50)
        self.pack_max_nei_b = config.get("hyper_parameters.pack_max_nei_b", 20)
        self.pack_max_nei_f = config.get("hyper_parameters.pack_max_nei_f", 20)

    def __getitem__(self, idx):
        data = self.data[idx]
        user, item, biz, friends, user_items, user_bizs, user_friends, user_packages, \
        pack_neighbors_b, pack_neighbors_f, label1, label2 = list(data.values())
        user = paddle.to_tensor(user)
        item = paddle.to_tensor(item)
        biz = paddle.to_tensor(biz)
        friends = paddle.to_tensor(
            sequence_padding_1d_list(friends, self.f_max_len))
        user_items = paddle.to_tensor(
            sequence_padding_1d_list(user_items[:273], self.u_max_i))
        user_bizs = paddle.to_tensor(
            sequence_padding_1d_list(user_bizs[:273], self.u_max_i))
        user_friends = paddle.to_tensor(
            sequence_padding_1d_list(user_friends[:289], self.u_max_f))
        user_packages = sequence_padding_2d(
            np.array(
                user_packages,
                dtype=np.int64).reshape([-1, self.f_max_len + 2])[:50],
            length=self.u_max_pack)
        user_packages = paddle.to_tensor(user_packages)
        pack_neighbors_b = sequence_padding_2d(
            np.array(
                pack_neighbors_b,
                dtype=np.int64).reshape([-1, self.f_max_len + 2]),
            length=self.pack_max_nei_b)
        pack_neighbors_b = paddle.to_tensor(pack_neighbors_b)
        pack_neighbors_f = sequence_padding_2d(
            np.array(
                pack_neighbors_f,
                dtype=np.int64).reshape([-1, self.f_max_len + 2]),
            length=self.pack_max_nei_f)
        pack_neighbors_f = paddle.to_tensor(pack_neighbors_f)
        label1 = paddle.to_tensor(label1)
        return (user, item, biz, friends, user_items, user_bizs, user_friends,
                user_packages, pack_neighbors_b, pack_neighbors_f, label1)

    def __len__(self):
        return len(self.data)


def sequence_padding_1d_list(inputs, length):
    return inputs[:length] + [0] * (length - len(inputs))


def sequence_padding_2d(inputs, length):
    padding_width = [[0, length - inputs.shape[0]], [0, 0]]
    inputs = np.pad(inputs[:length],
                    padding_width,
                    'constant',
                    constant_values=0)
    return inputs
