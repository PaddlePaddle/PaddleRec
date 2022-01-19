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
from random import shuffle
from scipy import sparse

from collections import defaultdict
from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config, test_list=''):
        super(RecDataset, self).__init__()
        self.config = config
        self.file_list = file_list
        self.test_list = test_list
        self.train_data = defaultdict(list)
        self.test_data = defaultdict(list)
        self.mode = config.get('runner.mode', 'test')
        self.init()

    def init(self):
        self.batch_size = self.config.get("runner.reader_batch_size", 1)
        user_id_map = defaultdict(int)
        item_id_map = defaultdict(int)

        u_id = 0
        i_id = 0
        for source_file in self.file_list:
            with open(source_file, 'r') as src:
                for line in src.readlines():
                    parts = line.strip().split('\t')
                    u_id_orig = int(parts[0])
                    if u_id_orig not in user_id_map:
                        user_id_map[u_id_orig] = u_id
                        u_id += 1

                    i_id_orig = int(parts[1])
                    if i_id_orig not in item_id_map:
                        item_id_map[i_id_orig] = i_id
                        i_id += 1

        major_map = user_id_map
        minor_map = item_id_map
        for source_file in self.file_list:
            with open(source_file, 'r') as src:
                for line in src.readlines():
                    parts = line.strip().split('\t')
                    key = major_map[int(parts[0])]
                    value = minor_map[int(parts[1])]
                    rating = np.float32(parts[2])
                    self.train_data[key].append((value, rating))

        if self.test_list:
            for source_file in self.test_list:
                with open(source_file, 'r') as src:
                    for line in src.readlines():
                        parts = line.strip().split('\t')
                        key = major_map[int(parts[0])]
                        value = minor_map[int(parts[1])]
                        rating = np.float32(parts[2])
                        self.test_data[key].append((value, rating))

        self.vector_dim = len(minor_map)

    def __iter__(self):
        if self.mode == 'train':
            data = self.train_data
            keys = list(data.keys())
            shuffle(keys)
            s_ind = 0
            e_ind = self.batch_size
            while e_ind < len(keys):
                local_ind = 0
                inds1 = []
                inds2 = []
                vals = []
                for ind in range(s_ind, e_ind):
                    inds2 += [v[0] for v in data[keys[ind]]]
                    inds1 += [local_ind] * len([v[0] for v in data[keys[ind]]])
                    vals += [v[1] for v in data[keys[ind]]]
                    local_ind += 1

                mini_batch = sparse.coo_matrix(
                    (vals, (inds1, inds2)),
                    shape=[self.batch_size, self.vector_dim],
                    dtype=np.float32).toarray()
                s_ind += self.batch_size
                e_ind += self.batch_size
                yield mini_batch
        else:
            src_data = self.train_data
            test_data = self.test_data
            keys = list(test_data.keys())
            shuffle(keys)
            s_ind = 0
            while s_ind < len(keys):
                inds1 = [0] * len([v[0] for v in self.test_data[keys[s_ind]]])
                inds2 = [v[0] for v in self.test_data[keys[s_ind]]]
                vals = [v[1] for v in test_data[keys[s_ind]]]

                src_inds1 = [0] * len([v[0] for v in src_data[keys[s_ind]]])
                src_inds2 = [v[0] for v in src_data[keys[s_ind]]]
                src_vals = [v[1] for v in src_data[keys[s_ind]]]

                mini_batch = (sparse.coo_matrix(
                    (vals, (inds1, inds2)),
                    shape=[1, self.vector_dim],
                    dtype=np.float32).toarray(), sparse.coo_matrix(
                        (src_vals, (src_inds1, src_inds2)),
                        shape=[1, self.vector_dim],
                        dtype=np.float32).toarray())
                s_ind += 1
                yield mini_batch
