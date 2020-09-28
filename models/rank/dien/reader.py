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

import os
import random

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from paddlerec.core.reader import ReaderBase
from paddlerec.core.utils import envs


class Reader(ReaderBase):
    def init(self):
        self.train_data_path = envs.get_global_env(
            "dataset.sample_1.data_path", None)
        self.res = []
        self.max_len = 0
        self.neg_candidate_item = []
        self.neg_candidate_cat = []
        self.max_neg_item = 10000
        self.max_neg_cat = 1000

        data_file_list = os.listdir(self.train_data_path)
        for i in range(0, len(data_file_list)):
            train_data_file = os.path.join(self.train_data_path,
                                           data_file_list[i])
            with open(train_data_file, "r") as fin:
                for line in fin:
                    line = line.strip().split(';')
                    hist = line[0].split()
                    self.max_len = max(self.max_len, len(hist))
        fo = open("tmp.txt", "w")
        fo.write(str(self.max_len))
        fo.close()
        self.batch_size = envs.get_global_env("dataset.sample_1.batch_size",
                                              32, None)
        self.group_size = self.batch_size * 20

    def _process_line(self, line):
        line = line.strip().split(';')
        hist = line[0].split()
        hist = [int(i) for i in hist]
        cate = line[1].split()
        cate = [int(i) for i in cate]
        return [hist, cate, [int(line[2])], [int(line[3])], [float(line[4])]]

    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """

        def data_iter():
            # feat_idx, feat_value, label = self._process_line(line)
            yield self._process_line(line)

        return data_iter

    def pad_batch_data(self, input, max_len):
        res = np.array([x + [0] * (max_len - len(x)) for x in input])
        res = res.astype("int64").reshape([-1, max_len])
        return res

    def make_data(self, b):
        max_len = max(len(x[0]) for x in b)
        # item = self.pad_batch_data([x[0] for x in b], max_len)
        # cat = self.pad_batch_data([x[1] for x in b], max_len)
        item = [x[0] for x in b]
        cat = [x[1] for x in b]
        neg_item = [None] * len(item)
        neg_cat = [None] * len(cat)

        for i in range(len(b)):
            neg_item[i] = []
            neg_cat[i] = []
            # Neg item and neg cat should be paried
            if len(self.neg_candidate_item) < self.max_neg_item:
                self.neg_candidate_item.extend(b[i][0])
                self.neg_candidate_cat.extend(b[i][1])
                if len(self.neg_candidate_item) > self.max_neg_item:
                    self.neg_candidate_item = self.neg_candidate_item[
                        0:self.max_neg_item]
                    self.neg_candidate_cat = self.neg_candidate_cat[
                        0:self.max_neg_item]
            else:
                len_seq = len(b[i][0])
                start_idx = random.randint(0, self.max_neg_item - len_seq - 1)
                self.neg_candidate_item[start_idx:start_idx + len_seq +
                                        1] = b[i][0]
                self.neg_candidate_cat[start_idx:start_idx + len_seq +
                                       1] = b[i][1]

            for _ in range(len(b[i][0])):
                randindex = random.randint(0, len(self.neg_candidate_item) - 1)
                neg_item[i].append(self.neg_candidate_item[randindex])
                neg_cat[i].append(self.neg_candidate_cat[randindex])

        len_array = [len(x[0]) for x in b]
        mask = np.array(
            [[0] * x + [-1e9] * (max_len - x) for x in len_array]).reshape(
                [-1, max_len, 1])
        target_item_seq = np.array(
            [[x[2]] * max_len for x in b]).astype("int64").reshape(
                [-1, max_len])
        target_cat_seq = np.array(
            [[x[3]] * max_len for x in b]).astype("int64").reshape(
                [-1, max_len])
        res = []
        for i in range(len(b)):
            res.append([
                item[i], cat[i], b[i][2], b[i][3], b[i][4], mask[i],
                target_item_seq[i], target_cat_seq[i], neg_item[i], neg_cat[i]
            ])
        return res

    def batch_reader(self, reader, batch_size, group_size):
        def batch_reader():
            bg = []
            for line in reader:
                bg.append(line)
                if len(bg) == group_size:
                    sortb = sorted(bg, key=lambda x: len(x[0]), reverse=False)
                    bg = []
                    for i in range(0, group_size, batch_size):
                        b = sortb[i:i + batch_size]
                        yield self.make_data(b)
            len_bg = len(bg)
            if len_bg != 0:
                sortb = sorted(bg, key=lambda x: len(x[0]), reverse=False)
                bg = []
                remain = len_bg % batch_size
                for i in range(0, len_bg - remain, batch_size):
                    b = sortb[i:i + batch_size]
                    yield self.make_data(b)

        return batch_reader

    def base_read(self, file_dir):
        res = []
        for train_file in file_dir:
            with open(train_file, "r") as fin:
                for line in fin:
                    line = line.strip().split(';')
                    hist = line[0].split()
                    cate = line[1].split()
                    res.append([hist, cate, line[2], line[3], float(line[4])])
        return res

    def generate_batch_from_trainfiles(self, files):
        data_set = self.base_read(files)
        random.shuffle(data_set)
        return self.batch_reader(data_set, self.batch_size,
                                 self.batch_size * 20)
