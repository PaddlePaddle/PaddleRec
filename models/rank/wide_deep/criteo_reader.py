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
import paddle
from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        if config:
            use_fleet = config.get("runner.use_fleet", False)
            self.inference = config.get("runner.inference", False)
        else:
            use_fleet = False
        if use_fleet:
            worker_id = paddle.distributed.get_rank()
            worker_num = paddle.distributed.get_world_size()
            file_num = len(file_list)
            if file_num < worker_num:
                raise ValueError(
                    "The number of data files is less than the number of workers"
                )
            blocksize = int(file_num / worker_num)
            self.file_list = file_list[worker_id * blocksize:(worker_id + 1) *
                                       blocksize]
            remainder = file_num - (blocksize * worker_num)
            if worker_id < remainder:
                self.file_list.append(file_list[-(worker_id + 1)])
        self.init()

    def init(self):
        padding = 0
        sparse_slots = "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
        self.sparse_slots = sparse_slots.strip().split(" ")
        self.dense_slots = ["dense_feature"]
        self.dense_slots_shape = [13]
        self.slots = self.sparse_slots + self.dense_slots
        self.slot2index = {}
        self.visit = {}
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding

    def __iter__(self):
        full_lines = []
        self.data = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    line = l.strip().split(" ")
                    output = [(i, []) for i in self.slots]
                    for i in line:
                        slot_feasign = i.split(":")
                        slot = slot_feasign[0]
                        if slot not in self.slots:
                            continue
                        if slot in self.sparse_slots:
                            feasign = int(slot_feasign[1])
                        else:
                            feasign = float(slot_feasign[1])
                        output[self.slot2index[slot]][1].append(feasign)
                        self.visit[slot] = True
                    for i in self.visit:
                        slot = i
                        if not self.visit[slot]:
                            if i in self.dense_slots:
                                output[self.slot2index[i]][1].extend(
                                    [self.padding] *
                                    self.dense_slots_shape[0])
                            else:
                                output[self.slot2index[i]][1].extend(
                                    [self.padding])
                        else:
                            self.visit[slot] = False
                    # sparse
                    output_list = []
                    for key, value in output[:-1]:
                        output_list.append(np.array(value).astype('int64'))
                    # dense
                    output_list.append(
                        np.array(output[-1][1]).astype("float32"))
                    # list
                    if self.inference:
                        yield output_list[1:]
                    else:
                        yield output_list
