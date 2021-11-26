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

from __future__ import print_function
import numpy as np
from paddle.io import IterableDataset
import random


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.maxlen = config.get("hyper_parameters.maxlen", 30)
        self.batch_size = config.get("runner.train_batch_size", 128)
        self.batches_per_epoch = config.get("runner.batches_per_epoch", 1000)
        self.init()
        self.count = 0

    def init(self):
        self.graph = {}
        self.users = set()
        self.items = set()
        for file in self.file_list:
            with open(file, "r") as f:
                for line in f:
                    conts = line.strip().split(',')
                    user_id = int(conts[0])
                    item_id = int(conts[1])
                    time_stamp = int(conts[2])
                    self.users.add(user_id)
                    self.items.add(item_id)
                    if user_id not in self.graph:
                        self.graph[user_id] = []
                    self.graph[user_id].append((item_id, time_stamp))
        for user_id, value in self.graph.items():
            value.sort(key=lambda x: x[1])
            self.graph[user_id] = [x[0] for x in value]
        self.users = list(self.users)
        self.items = list(self.items)

    def __iter__(self):
        random.seed(12345)
        while True:
            user_id_list = random.sample(self.users, self.batch_size)
            if self.count >= self.batches_per_epoch * self.batch_size:
                self.count = 0
                break
            for user_id in user_id_list:
                item_list = self.graph[user_id]
                if len(item_list) <= 4:
                    continue
                k = random.choice(range(4, len(item_list)))
                item_id = item_list[k]

                if k >= self.maxlen:
                    hist_item_list = item_list[k - self.maxlen:k]
                    hist_item_len = len(hist_item_list)
                else:
                    hist_item_list = item_list[:k] + [0] * (self.maxlen - k)
                    hist_item_len = k
                self.count += 1
                yield [
                    np.array(hist_item_list).astype("int64"),
                    np.array([item_id]).astype("int64"),
                    np.array([hist_item_len]).astype("int64")
                ]
