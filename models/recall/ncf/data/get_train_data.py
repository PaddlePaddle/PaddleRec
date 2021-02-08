# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import scipy.sparse as sp
import numpy as np
from time import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument(
        '--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument(
        '--dataset', nargs='?', default='ml-1m', help='Choose a dataset.')
    parser.add_argument(
        '--num_neg',
        type=int,
        default=4,
        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument(
        '--train_data_path',
        type=str,
        default="Data/train_data.csv",
        help='train_data_path')
    return parser.parse_args()


def get_train_data(filename, write_file, num_negatives):
    '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
    # Get number of users and items
    num_users, num_items = 0, 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
    print("users_num:", num_users, "items_num:", num_items)
    # Construct matrix
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if (rating > 0):
                mat[user, item] = 1.0
            line = f.readline()

    file = open(write_file, 'w')
    print("writing " + write_file)

    for (u, i) in mat.keys():
        # positive instance
        user_input = str(u)
        item_input = str(i)
        label = str(1)
        sample = "{0},{1},{2}".format(user_input, item_input, label) + "\n"
        file.write(sample)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in mat.keys():
                j = np.random.randint(num_items)
            user_input = str(u)
            item_input = str(j)
            label = str(0)
            sample = "{0},{1},{2}".format(user_input, item_input, label) + "\n"
            file.write(sample)


if __name__ == "__main__":
    args = parse_args()
    get_train_data(args.path + args.dataset + ".train.rating",
                   args.train_data_path, args.num_neg)
