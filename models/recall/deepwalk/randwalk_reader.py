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

import os

import paddle
import numpy as np

from pgl import graph_kernel
from pgl.utils.logger import log
from pgl.utils.data import Dataset
from pgl.sampling import random_walk
from pgl.graph_kernel import skip_gram_gen_pair


class BatchRandWalk(object):
    def __init__(self, graph, walk_len, win_size, neg_num, neg_sample_type):
        self.graph = graph
        self.walk_len = walk_len
        self.win_size = win_size
        self.neg_num = neg_num
        self.neg_sample_type = neg_sample_type

    def __call__(self, nodes):
        walks = random_walk(self.graph, nodes, self.walk_len)
        src_list, pos_list = [], []
        for walk in walks:
            s, p = skip_gram_gen_pair(walk, self.win_size)
            src_list.append(s), pos_list.append(p)
        src = [s for x in src_list for s in x]
        pos = [s for x in pos_list for s in x]
        src = np.array(src, dtype=np.int64),
        pos = np.array(pos, dtype=np.int64)
        src, pos = np.reshape(src, [-1, 1]), np.reshape(pos, [-1, 1])

        neg_sample_size = [len(pos), self.neg_num]
        if self.neg_sample_type == "average":
            negs = np.random.randint(
                low=0, high=self.graph.num_nodes, size=neg_sample_size)
        elif self.neg_sample_type == "outdegree":
            pass
            #negs = alias_sample(neg_sample_size, alias, events)
        elif self.neg_sample_type == "inbatch":
            pass
        else:
            raise ValueError
        dsts = np.concatenate([pos, negs], 1)
        # [batch_size, 1] [batch_size, neg_num+1]
        return src, dsts


class ShardedDataset(Dataset):
    def __init__(self, nodes, mode="train", repeat=1):
        self.repeat = repeat
        if int(paddle.distributed.get_world_size()) == 1 or mode != "train":
            self.data = nodes
        else:
            self.data = nodes[int(paddle.distributed.get_rank())::int(
                paddle.distributed.get_world_size())]

    def __getitem__(self, idx):
        return self.data[idx % len(self.data)]

    def __len__(self):
        return len(self.data) * self.repeat
