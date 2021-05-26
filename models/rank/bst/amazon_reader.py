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
import random
from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.config = config
        self.init()

    def init(self):
        from operator import mul
        padding = 0
        sparse_slots = "label userid history cate position target target_cate target_position"
        self.sparse_slots = sparse_slots.strip('\n').strip().split(" ")
        self.slots = self.sparse_slots
        self.slot2index = {}
        self.visit = {}
        self.batch_size = self.config.get("runner.train_batch_size")
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding

    def __iter__(self):
        full_lines = []
        self.data = []
        data_set = []
        max_len = 0
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    data_set.append(l)
                    line = l.strip().split(" ")
                    len_hist = 0
                    for i in line:
                        if i.split(':')[0] == 'history':
                            len_hist = len_hist + 1
                    max_len = max(max_len, len_hist)

        for l in data_set:
            line = l.strip().split(" ")
            output = [(i, []) for i in self.slots]
            for i in line:
                slot_feasign = i.split(":")
                slot = slot_feasign[0]
                if slot not in self.slots:
                    continue
                feasign = int(slot_feasign[1])
                output[self.slot2index[slot]][1].append(feasign)
                self.visit[slot] = True
            for i in self.visit:
                slot = i
                if not self.visit[slot]:
                    output[self.slot2index[i]][1].extend([self.padding])
                else:
                    self.visit[slot] = False
            # sparse
            output_list = []
            #for key, value in output[:-1]:
            for i in range(len(output)):
                if i == 2 or i == 3 or i == 4:
                    output_list.append(
                        np.array(output[i][1] + [0] * (max_len - len(output[i][
                            1]))).astype('int64'))
                else:
                    output_list.append(np.array(output[i][1]).astype('int64'))
            # dense
            yield output_list
