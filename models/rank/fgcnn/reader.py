#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.io import IterableDataset
import h5py


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list

    def __iter__(self):
        self.data = []
        for file in self.file_list:
            print(file)
            f = h5py.File(file, 'r')
            key = list(f.keys())[0]
            for l in f[key]:
                output_list = []
                output_list.append(l[0:39].astype('int64'))
                output_list.append(l[39:].astype('int64'))
                yield output_list
