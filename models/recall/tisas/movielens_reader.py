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
import os
import scipy.sparse
from paddle.io import Dataset

import sys
import copy
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import paddle


class RecDataset(Dataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        seed = config.get("hyper_parameters.seed")
        np.random.seed(2021)
        self.user_train, self.user_valid, self.user_test, self.usernum, self.itemnum, timenum = data_partition(
            file_list[0])
        self.maxlen = config.get("hyper_parameters.maxlen")
        self.time_span = config.get("hyper_parameters.time_span")
        self.relation_matrix = Relation(self.user_train, self.usernum,
                                        self.maxlen, self.time_span)
        self.mode = config.get("runner.mode", 'test')
        if self.mode == 'test':
            self.test_users = []
            user_idx = range(1, self.usernum + 1)
            if self.usernum > 10000:
                user_idx = np.random.sample(user_idx, 10000)

            self.test_users = [
                u for u in user_idx
                if len(self.user_train[u]) > 0 or len(self.user_train[u]) > 0
            ]

    def __getitem__(self, idx):
        if self.mode == 'train':
            user = np.random.randint(1, self.usernum + 1)
            while len(self.user_train[user]) <= 1:
                user = np.random.randint(1, self.usernum + 1)
            return self.sample(user)

        else:
            u = self.test_users[idx]
            seq = np.zeros([self.maxlen], dtype=np.int64)
            time_seq = np.zeros([self.maxlen], dtype=np.int64)
            idx = self.maxlen - 1
            seq[idx] = self.user_valid[u][0][0]
            time_seq[idx] = self.user_valid[u][0][1]
            idx -= 1
            for i in reversed(self.user_train[u]):
                seq[idx] = i[0]
                time_seq[idx] = i[1]
                idx -= 1
                if idx == -1:
                    break
            time_matrix = computeRePos(time_seq, self.time_span)
            item_idx = [self.user_test[u][0][0]]
            rated = set(map(lambda x: x[0], self.user_train[u]))
            rated.add(self.user_valid[u][0][0])
            rated.add(self.user_valid[u][0][0])
            rated.add(0)
            for _ in range(100):
                t = np.random.randint(1, self.itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, self.itemnum + 1)
                item_idx.append(t)
            return seq, time_matrix, np.array(item_idx)

    def __len__(self):
        if self.mode == 'train':
            return len(self.user_train)
        else:
            return len(self.test_users)

    def sample(self, user):
        seq = np.zeros([self.maxlen], dtype=np.int64)
        time_seq = np.zeros([self.maxlen], dtype=np.int64)
        pos = np.zeros([self.maxlen], dtype=np.int64)
        neg = np.zeros([self.maxlen], dtype=np.int64)
        nxt = self.user_train[user][-1][0]

        idx = self.maxlen - 1
        ts = set(map(lambda x: x[0], self.user_train[user]))
        for i in reversed(self.user_train[user][:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1:
                break
        time_matrix = self.relation_matrix[user].astype('int64')
        return seq, time_matrix, pos, neg


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def computeRePos(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def Relation(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(range(1, usernum + 1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:  # float as map key?
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(
            map(lambda x: [item_map[x[0]], time_map[x[1]]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1)], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)


def data_partition(fname):
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    f = open(fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open(fname, 'r')  # try?...ugly data pre-processing code
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        if user_count[u] < 5 or item_count[i] < 5:  # hard-coded
            continue
        time_set.add(timestamp)
        User[u].append([i, timestamp])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum, timenum]
