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


class RecDataset(Dataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        data_root = os.path.dirname(file_list[0])
        self.data = LoadData(data_root)
        self.mode = config.get('runner.mode', 'test')

    def __getitem__(self, idx):
        if self.mode == 'train':
            return (np.array(self.data.user_train[idx]),
                    np.array(self.data.item_map_list),
                    np.array(self.data.item_train[idx]),
                    np.array(self.data.item_bind_M))
        else:
            return (np.array(self.data.user_test[idx]),
                    np.array(self.data.item_map_list))

    def __len__(self):
        if self.mode == 'train':
            return len(self.data.user_train)
        return len(self.data.user_test)


class LoadData(object):
    def __init__(self, DATA_ROOT):
        self.trainfile = os.path.join(DATA_ROOT, 'train.csv')
        self.testfile = os.path.join(DATA_ROOT, 'test.csv')
        self.user_field_M, self.item_field_M = self.get_length()
        self.item_bind_M = self.bind_item()  # assaign a userID for a specific user-context
        self.user_bind_M = self.bind_user()  # assaign a itemID for a specific item-feature
        self.item_map_list = []
        for itemid in self.item_map.keys():
            self.item_map_list.append([int(feature) for feature in self.item_map[itemid].strip().split('-')[0:]])
        self.item_map_list.append([int(feature) for feature in self.item_map[0].strip().split('-')[0:]])
        self.user_positive_list = self.get_positive_list(self.trainfile)  # userID positive itemID
        self.Train_data, self.Test_data = self.construct_data()
        self.user_train, self.item_train = self.get_train_instances()
        self.user_test = self.get_test()

    def get_length(self):
        '''
        map the user fields in all files, kept in self.user_fields dictionary
        :return:
        '''
        length_user = 0
        length_item = 0
        length_item = 0
        f = open(self.trainfile)
        line = f.readline()
        while line:
            user_features = line.strip().split(',')[0].split('-')
            item_features = line.strip().split(',')[1].split('-')
            for user_feature in user_features:
                feature = int(user_feature)
                if feature > length_user:
                    length_user = feature
            for item_feature in item_features:
                feature = int(item_feature)
                if feature > length_item:
                    length_item = feature
            line = f.readline()
        f.close()
        return length_user + 1, length_item + 1

    def bind_item(self):
        '''
        Bind item and feature
        :return:
        '''
        self.binded_items = {}  # dic{feature: id}
        self.item_map = {}  # dic{id: feature}
        self.bind_i(self.trainfile)
        self.bind_i(self.testfile)
        return len(self.binded_items)

    def bind_i(self, file):
        '''
        Read a feature file and bind
        :param file: feature file
        :return:
        '''
        f = open(file)
        line = f.readline()
        i = len(self.binded_items)
        while line:
            features = line.strip().split(',')
            item_features = features[1]
            if item_features not in self.binded_items:
                self.binded_items[item_features] = i
                self.item_map[i] = item_features
                i = i + 1
            line = f.readline()
        f.close()

    def bind_user(self):
        '''
        Map the item fields in all files, kept in self.item_fields dictionary
        :return:bind_user
        '''
        self.binded_users = {}
        self.user_map = {}
        self.bind_u(self.trainfile)
        self.bind_u(self.testfile)
        return len(self.binded_users)

    def bind_u(self, file):
        '''
        Read a feature file and bind
        :param file:
        :return:
        '''
        f = open(file)
        line = f.readline()
        i = len(self.binded_users)
        while line:
            features = line.strip().split(',')
            user_features = features[0]
            if user_features not in self.binded_users:
                self.binded_users[user_features] = i
                self.user_map[i] = user_features
                i = i + 1
            line = f.readline()
        f.close()

    def get_positive_list(self, file):
        '''
        Obtain positive item lists for each user
        :param file: train file
        :return:
        '''
        self.max_positive_len = 0
        f = open(file)
        line = f.readline()
        user_positive_list = {}
        while line:
            features = line.strip().split(',')
            user_id = self.binded_users[features[0]]
            item_id = self.binded_items[features[1]]
            if user_id in user_positive_list:
                user_positive_list[user_id].append(item_id)
            else:
                user_positive_list[user_id] = [item_id]
            line = f.readline()
        f.close()
        for i in user_positive_list:
            if len(user_positive_list[i]) > self.max_positive_len:
                self.max_positive_len = len(user_positive_list[i])
        return user_positive_list

    def get_train_instances(self):
        user_train, item_train = [], []
        for i in self.user_positive_list:
            u_train = [int(feature) for feature in self.user_map[i].strip().split('-')[0:]]
            user_train.append(u_train)
            temp = self.user_positive_list[i]
            while len(temp) < self.max_positive_len:
                temp.append(self.item_bind_M)
            item_train.append(temp)
        user_train = np.array(user_train)
        item_train = np.array(item_train)
        return user_train, item_train

    def construct_data(self):
        X_user, X_item = self.read_data(self.trainfile)
        Train_data = self.construct_dataset(X_user, X_item)
        print("# of training:", len(X_user))
        X_user, X_item = self.read_data(self.testfile)
        Test_data = self.construct_dataset(X_user, X_item)
        print("# of test:", len(X_user))

        return Train_data, Test_data

    def construct_dataset(self, X_user, X_item):

        user_id = []
        for one in X_user:
            user_id.append(self.binded_users["-".join([str(item) for item in one[0:]])])
        item_id = []
        for one in X_item:
            item_id.append(self.binded_items["-".join([str(item) for item in one[0:]])])
        count = np.ones(len(X_user))
        sparse_matrix = scipy.sparse.csr_matrix((count, (user_id, item_id)), dtype=np.int16,
                                                shape=(self.user_bind_M, self.item_bind_M))
        return sparse_matrix

    def get_test(self):
        X_user, X_item = self.read_data(self.testfile)
        return X_user

    # lists of user and item
    def read_data(self, file):
        '''
        read raw data
        :param file: data file
        :return: structured data
        '''
        # read a data file;
        f = open(file)
        X_user = []
        X_item = []
        line = f.readline()
        while line:
            features = line.strip().split(',')
            user_features = features[0].split('-')
            X_user.append([int(item) for item in user_features[0:]])
            item_features = features[1].split('-')
            X_item.append([int(item) for item in item_features[0:]])
            line = f.readline()
        f.close()
        return X_user, X_item
