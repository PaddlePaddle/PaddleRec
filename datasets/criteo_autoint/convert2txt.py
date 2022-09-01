# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import argparse

parser = argparse.ArgumentParser('convert npy to txt file')
parser.add_argument(
    '--root_dir',
    type=str,
    default='./Criteo',
    required=False,
    help='root directory of src data')
args = parser.parse_args()


def write_to_file(output_folder, file_path_list):
    fmt_str = ['%d'] + ['%d'] * 39 + ['%.7f'] * 39
    for folder in file_path_list:
        if not os.path.isdir(folder): continue
        print("begin {}".format(folder))
        feature_index = np.load(
            os.path.join(folder, 'train_i.npy'),
            allow_pickle=True).astype('int64')
        feature_value = np.load(
            os.path.join(folder, 'train_x2.npy'),
            allow_pickle=True).astype('float32')
        label = np.load(
            os.path.join(folder, 'train_y.npy'),
            allow_pickle=True).astype('int64').reshape([-1, 1])
        data = np.concatenate((label, feature_index, feature_value), axis=1)
        np.savetxt(
            os.path.join(output_folder, os.path.basename(folder)),
            data,
            fmt=' '.join(fmt_str))
        print("complete {}".format(folder))


if __name__ == '__main__':
    train_folders = [
        os.path.join(args.root_dir, 'part{}'.format(i)) for i in range(3, 11)
    ]
    test_folders = [os.path.join(args.root_dir, 'part1')]
    write_to_file('./slot_test_data_full', test_folders)
    write_to_file('./slot_train_data_full', train_folders)
