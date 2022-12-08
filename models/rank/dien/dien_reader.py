#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import numpy as np
import io
import random
import paddle
from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.config = config
        self.init()
        self.item_count = config.get("hyper_parameters.item_count")

    def init(self):
        self.res = []
        self.max_len = 0
        self.neg_candidate_item = []
        self.neg_candidate_cat = []
        self.max_neg_item = 10000
        self.max_neg_cat = 1000

        for file in self.file_list:
            with open(file, "r") as fin:
                for line in fin:
                    line = line.strip().split(';')
                    hist = line[0].split()
                    self.max_len = max(self.max_len, len(hist))
        fo = open("tmp.txt", "w")
        fo.write(str(self.max_len))
        fo.close()
        self.batch_size = self.config.get("runner.train_batch_size")
        self.item_count = self.config.get("hyper_parameters.item_count", 63001)
        self.cat_count = self.config.get("hyper_parameters.cat_count", 801)
        self.group_size = (self.batch_size) * 20

    def __iter__(self):
        file_dir = self.file_list
        res0 = []
        for train_file in file_dir:
            with open(train_file, "r") as fin:
                for line in fin:
                    line = line.strip().split(';')
                    if len(line) != 5:
                        continue
                    hist = line[0].split()
                    tmp = [int(x) for x in hist]
                    if max(tmp) > self.item_count:
                        continue
                    cate = line[1].split()
                    tmp = [int(x) for x in cate]
                    if max(tmp) > self.cat_count:
                        continue
                    res0.append([hist, cate, line[2], line[3], float(line[4])])

        data_set = res0
        random.seed(12345)
        random.shuffle(data_set)

        reader, batch_size, group_size = data_set, self.batch_size, self.group_size
        bg = []
        for line in reader:
            bg.append(line)
            if len(bg) == group_size:
                sortb = sorted(bg, key=lambda x: len(x[0]), reverse=False)

                bg = []
                for i in range(0, group_size, batch_size):
                    b = sortb[i:i + batch_size]
                    max_len = max(len(x[0]) for x in b)
                    if max_len < 2:
                        continue

                    itemInput = [x[0] for x in b]
                    itemRes0 = np.array(
                        [x + [0] * (max_len - len(x)) for x in itemInput])
                    item = itemRes0.astype("int64").reshape([-1, max_len])
                    catInput = [x[1] for x in b]
                    catRes0 = np.array(
                        [x + [0] * (max_len - len(x)) for x in catInput])
                    cat = catRes0.astype("int64").reshape([-1, max_len])

                    len_array = [len(x[0]) for x in b]
                    mask = np.array(
                        [[0] * x + [-1e9] * (max_len - x)
                         for x in len_array]).reshape([-1, max_len, 1])
                    target_item_seq = np.array(
                        [[x[2]] * max_len for x in b]).astype("int64").reshape(
                            [-1, max_len])
                    target_cat_seq = np.array(
                        [[x[3]] * max_len for x in b]).astype("int64").reshape(
                            [-1, max_len])

                    neg_item = [None] * len(item)
                    neg_cat = [None] * len(cat)

                    for i in range(len(b)):
                        neg_item[i] = []
                        neg_cat[i] = []
                        if len(self.neg_candidate_item) < self.max_neg_item:
                            self.neg_candidate_item.extend(b[i][0])
                            if len(self.
                                   neg_candidate_item) > self.max_neg_item:
                                self.neg_candidate_item = self.neg_candidate_item[
                                    0:self.max_neg_item]
                        else:
                            len_seq = len(b[i][0])
                            start_idx = random.randint(
                                0, self.max_neg_item - len_seq - 1)
                            self.neg_candidate_item[start_idx:start_idx +
                                                    len_seq + 1] = b[i][0]

                        if len(self.neg_candidate_cat) < self.max_neg_cat:
                            self.neg_candidate_cat.extend(b[i][1])
                            if len(self.neg_candidate_cat) > self.max_neg_cat:
                                self.neg_candidate_cat = self.neg_candidate_cat[
                                    0:self.max_neg_cat]
                        else:
                            len_seq = len(b[i][1])
                            start_idx = random.randint(
                                0, self.max_neg_cat - len_seq - 1)
                            self.neg_candidate_item[start_idx:start_idx +
                                                    len_seq + 1] = b[i][1]
                        for _ in range(max_len):
                            neg_item[i].append(self.neg_candidate_item[
                                random.randint(
                                    0, len(self.neg_candidate_item) - 1)])
                        for _ in range(max_len):
                            neg_cat[i].append(self.neg_candidate_cat[
                                random.randint(
                                    0, len(self.neg_candidate_cat) - 1)])

                    for i in range(len(b)):
                        res = []
                        # res0 = []
                        res.append(np.array(item[i]))
                        res.append(np.array(cat[i]))
                        res.append(np.array(b[i][2]).astype('int64'))
                        res.append(np.array(b[i][3]).astype('int64'))
                        res.append(np.array(b[i][4]).astype('float32'))
                        res.append(np.array(mask[i]).astype('float32'))
                        res.append(np.array(target_item_seq[i]))
                        res.append(np.array(target_cat_seq[i]).astype('int64'))
                        res.append(np.array(neg_item[i]).astype('int64'))
                        res.append(np.array(neg_cat[i]).astype('int64'))

                        yield res

        len_bg = len(bg)
        if len_bg != 0:
            sortb = sorted(bg, key=lambda x: len(x[0]), reverse=False)
            bg = []
            remain = len_bg % batch_size
            for i in range(0, len_bg - remain, batch_size):
                b = sortb[i:i + batch_size]

                max_len = max(len(x[0]) for x in b)
                if max_len < 2: continue
                itemInput = [x[0] for x in b]
                itemRes0 = np.array(
                    [x + [0] * (max_len - len(x)) for x in itemInput])
                item = itemRes0.astype("int64").reshape([-1, max_len])
                catInput = [x[1] for x in b]
                catRes0 = np.array(
                    [x + [0] * (max_len - len(x)) for x in catInput])
                cat = catRes0.astype("int64").reshape([-1, max_len])

                len_array = [len(x[0]) for x in b]
                mask = np.array(
                    [[0] * x + [-1e9] * (max_len - x)
                     for x in len_array]).reshape([-1, max_len, 1])
                target_item_seq = np.array(
                    [[x[2]] * max_len for x in b]).astype("int64").reshape(
                        [-1, max_len])
                target_cat_seq = np.array(
                    [[x[3]] * max_len for x in b]).astype("int64").reshape(
                        [-1, max_len])
                neg_item = [None] * len(item)
                neg_cat = [None] * len(cat)

                for i in range(len(b)):
                    neg_item[i] = []
                    neg_cat[i] = []
                    if len(self.neg_candidate_item) < self.max_neg_item:
                        self.neg_candidate_item.extend(b[i][0])
                        if len(self.neg_candidate_item) > self.max_neg_item:
                            self.neg_candidate_item = self.neg_candidate_item[
                                0:self.max_neg_item]
                    else:
                        len_seq = len(b[i][0])
                        start_idx = random.randint(
                            0, self.max_neg_item - len_seq - 1)
                        self.neg_candidate_item[start_idx:start_idx + len_seq +
                                                1] = b[i][0]

                    if len(self.neg_candidate_cat) < self.max_neg_cat:
                        self.neg_candidate_cat.extend(b[i][1])
                        if len(self.neg_candidate_cat) > self.max_neg_cat:
                            self.neg_candidate_cat = self.neg_candidate_cat[
                                0:self.max_neg_cat]
                    else:
                        len_seq = len(b[i][1])
                        start_idx = random.randint(
                            0, self.max_neg_cat - len_seq - 1)
                        self.neg_candidate_item[start_idx:start_idx + len_seq +
                                                1] = b[i][1]
                    for _ in range(max_len):
                        neg_item[i].append(self.neg_candidate_item[
                            random.randint(0, len(self.neg_candidate_item) -
                                           1)])
                    for _ in range(max_len):
                        neg_cat[i].append(self.neg_candidate_cat[
                            random.randint(0, len(self.neg_candidate_cat) -
                                           1)])

                for i in range(len(b)):
                    res = []
                    res.append(np.array(item[i]))
                    res.append(np.array(cat[i]))
                    res.append(np.array(b[i][2]).astype('int64'))
                    res.append(np.array(b[i][3]).astype('int64'))
                    res.append(np.array(b[i][4]).astype('float32'))
                    res.append(np.array(mask[i]).astype('float32'))
                    res.append(np.array(target_item_seq[i]))
                    res.append(np.array(target_cat_seq[i]))
                    res.append(np.array(neg_item[i]).astype('int64'))
                    res.append(np.array(neg_cat[i]).astype('int64'))
                    yield res
