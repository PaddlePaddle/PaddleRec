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
import os
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
        self.file_name = ['train_i.npy', 'train_x2.npy', 'train_y.npy']

    def __iter__(self):
        for file in self.file_list:
            with open(file, 'r') as rf:
                for l in rf:
                    line = l.strip().split(' ')

                    output_list = []
                    output_list.append(np.array([line[0]]).astype('int64'))
                    output_list.append(np.array(line[1:40]).astype('int64'))
                    output_list.append(np.array(line[40:]).astype('float32'))
                    if self.inference:
                        yield output_list[1:]
                    else:
                        yield output_list
