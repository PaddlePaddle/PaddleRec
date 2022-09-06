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
import pandas as pd
import scipy.sparse as sp
from paddle.io import IterableDataset
import random

random.seed(42)


def load_all(train_rating, test_negative, test_num=100):
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv(
        train_rating,
        sep='\t',
        header=None,
        names=['user', 'item'],
        usecols=[0, 1],
        dtype={0: np.int32,
               1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open(test_negative, 'r') as fd:
        line = fd.readline()
        while line is not None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()
    return train_data, test_data, user_num, item_num, train_mat


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.train_rating = config.get("runner.train_rating", None)
        self.test_negative = config.get("runner.test_negative", None)
        # 根据 file_list 中路径名判断是 "train" or "test"
        self.is_training = True if sum(
            ["train" in path for path in file_list]) > 0 else False
        train_data, test_data, user_num, item_num, train_mat = load_all(
            self.train_rating, self.test_negative)

        self.features_ps = train_data if self.is_training else test_data
        self.num_item = item_num
        self.num_ng = config.get("runner.num_ng",
                                 4) if self.is_training else 100
        self.train_mat = train_mat
        self.labels = [
            0 if i % self.num_ng else 1 for i in range(len(self.features_ps))
        ]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __iter__(self):
        if self.is_training:
            self.ng_sample()

        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels

        if self.is_training:
            temp = list(zip(features, labels))
            random.shuffle(temp)
            features, labels = zip(*temp)
            features, labels = list(features), list(labels)

        for idx in range(len(labels)):
            output_list = []
            user = features[idx][0]
            item = features[idx][1]
            label = labels[idx]
            output_list.append(np.array(user).astype('int64'))
            output_list.append(np.array(item).astype('int64'))
            output_list.append(np.array(label).astype('int64'))
            yield output_list
