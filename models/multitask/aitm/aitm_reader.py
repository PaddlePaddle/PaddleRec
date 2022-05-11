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
import paddle
from paddle.io import Dataset


class RecDataset(Dataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.feature_names = []
        self.datafile = file_list[0]
        self.data = []
        self._load_data()

    def _load_data(self):
        print("start load data from: {}".format(self.datafile))
        count = 0
        with open(self.datafile) as f:
            self.feature_names = f.readline().strip().split(',')[2:]
            for line in f:
                count += 1
                line = line.strip().split(',')
                line = [int(v) for v in line]
                self.data.append(line)
        print("load data from {} finished".format(self.datafile))

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        click = line[0]
        conversion = line[1]
        # features = dict(zip(self.feature_names, line[2:]))
        features = line[2:]
        return click, conversion, features
