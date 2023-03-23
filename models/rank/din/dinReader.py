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

    def init(self):
        self.res = []
        self.max_len = 0

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
        self.group_size = (self.batch_size) * 20

    def __iter__(self):
        file_dir = self.file_list
        res0 = []
        for train_file in file_dir:
            with open(train_file, "r") as fin:
                for line in fin:
                    line = line.strip().split(';')
                    if len(line) < 5:
                        continue
                    hist = line[0].split()
                    cate = line[1].split()
                    res0.append([hist, cate, line[2], line[3], float(line[4])])

        data_set = res0
        #random.shuffle(data_set)

        reader, batch_size, group_size = data_set, self.batch_size, self.group_size
        bg = []
        for line in reader:
            bg.append(line)
            if len(bg) == group_size:  # #
                sortb = sorted(bg, key=lambda x: len(x[0]), reverse=False)
                bg = []
                for i in range(0, group_size, batch_size):
                    b = sortb[i:i + batch_size]
                    max_len = max(len(x[0]) for x in b)

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

                    for i in range(len(b)):
                        res = []
                        res.append(np.array(item[i]))
                        res.append(np.array(cat[i]))
                        res.append(np.array(b[i][2]).astype('int64'))
                        res.append(np.array(b[i][3]).astype('int64'))
                        res.append(np.array(b[i][4]).astype('float32'))
                        res.append(np.array(mask[i]).astype('int64'))
                        res.append(np.array(target_item_seq[i]))
                        res.append(np.array(target_cat_seq[i]))
                        yield res

        len_bg = len(bg)
        if len_bg != 0:
            sortb = sorted(bg, key=lambda x: len(x[0]), reverse=False)
            bg = []
            remain = len_bg % batch_size
            for i in range(0, len_bg - remain, batch_size):
                b = sortb[i:i + batch_size]

                max_len = max(len(x[0]) for x in b)

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

                for i in range(len(b)):
                    res = []
                    res.append(np.array(item[i]))
                    res.append(np.array(cat[i]))
                    res.append(np.array(b[i][2]).astype('int64'))
                    res.append(np.array(b[i][3]).astype('int64'))
                    res.append(np.array(b[i][4]).astype('float32'))
                    res.append(np.array(mask[i]).astype('int64'))
                    res.append(np.array(target_item_seq[i]))
                    res.append(np.array(target_cat_seq[i]))
                    yield res
