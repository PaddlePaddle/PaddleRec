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


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.maxlen = config.get("hyper_parameters.maxlen", 30)
        self.init()

    def init(self):
        padding = 0
        sparse_slots = "hist_item eval_item"
        self.sparse_slots = sparse_slots.strip().split(" ")
        self.slots = self.sparse_slots
        self.slot2index = {}
        self.visit = {}
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding

    def __iter__(self):
        for file in self.file_list:
            with open(file, "r") as rf:
                for line in rf:
                    lines = line.strip().split(" ")
                    # print(lines)
                    output = [(i, []) for i in self.slots]
                    for i in lines:
                        slot_feasign = i.split(":")
                        slot = slot_feasign[0]
                        if slot not in self.slots:
                            continue
                        if slot in self.sparse_slots:
                            feasign = int(slot_feasign[1])
                            output[self.slot2index[slot]][1].append(feasign)
                    output_list = []
                    seq_lens = []
                    eval_list = []
                    for key, value in output:
                        if key == "hist_item":
                            seq_lens.append(min(self.maxlen, len(value)))
                            value = value[-self.maxlen:] + [self.padding] * \
                                max(0, self.maxlen - len(value))
                        if key == "eval_item":
                            value = value[:self.maxlen] + [self.padding] * \
                                max(0, self.maxlen - len(value))
                            eval_list.append(value)
                            continue
                        output_list.append(np.array(value).astype("int64"))
                    if len(eval_list) == 0:
                        continue
                    yield output_list + [
                        np.array(seq_lens).astype("int64")
                    ] + [np.array(eval_list).astype("int64")]
