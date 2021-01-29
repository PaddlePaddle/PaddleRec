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

from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list

    def __iter__(self):
        full_lines = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    l = l.strip().split(" ")
                    data = l[0:100]
                    #seq_len = l[100:101]
                    label = l[101:]
                    if data is None:
                        continue
                    data_tensor = np.array(
                        [int(i) for i in data]).astype('int64')
                    label_tensor = np.array(
                        [int(i) for i in label]).astype('int64')
                    yield data_tensor, label_tensor
