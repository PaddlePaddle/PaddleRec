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

from __future__ import print_function
import numpy as np
import os
from paddle.io import IterableDataset
import pandas as pd

sparse_features = [
    'userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id',
    'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level',
    'occupation', 'new_user_class_level ', 'campaign_id', 'customer',
    'cate_id', 'brand'
]

dense_features = ['price']


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super().__init__()
        self.file_list = file_list
        print(file_list)
        data_file = [f.split('/')[-1] for f in file_list]
        mode = data_file[0].split('_')[0]
        data_dir = file_list[0].split(data_file[0])[0]
        data_dir = data_dir[:-1]
        data_dir = os.path.join(data_dir, os.path.split(data_file[0])[0])
        assert (mode == 'train' or mode == 'test' or mode == 'sample'
                ), f"mode must be 'train' or 'test', but get '{mode}'"
        feat_input = pd.read_pickle(
            os.path.join(data_dir, mode + '_feat_input.pkl'))
        self.sess_input = pd.read_pickle(
            os.path.join(data_dir, mode + '_sess_input.pkl'))
        self.sess_length = pd.read_pickle(
            os.path.join(data_dir, mode + '_session_length.pkl'))
        self.label = pd.read_pickle(
            os.path.join(data_dir, mode + '_label.pkl'))
        if str(type(self.label)).split("'")[1] != 'numpy.ndarray':
            self.label = self.label.to_numpy()
        self.label = self.label.astype('int64')
        self.num_samples = self.label.shape[0]
        self.sparse_input = feat_input[sparse_features].to_numpy().astype(
            'int64')
        self.dense_input = feat_input[dense_features].to_numpy().reshape(
            -1).astype('float32')

    def __iter__(self):
        for i in range(self.num_samples):
            yield [
                self.sparse_input[i, :], self.dense_input[i],
                self.sess_input[i, :, :], self.sess_length[i], self.label[i]
            ]
