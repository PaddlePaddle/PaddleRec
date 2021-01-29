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
        self.config = config

    def __iter__(self):
        full_lines = []
        self.data = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    l = l.strip().split(',')
                    l = list(map(float, l))
                    label_income = []
                    label_marital = []
                    data = l[2:]
                    if int(l[1]) == 0:
                        label_income = [0]
                    elif int(l[1]) == 1:
                        label_income = [1]
                    if int(l[0]) == 0:
                        label_marital = [0]
                    elif int(l[0]) == 1:
                        label_marital = [1]
                    output_list = []
                    output_list.append(np.array(data).astype('float32'))
                    output_list.append(np.array(label_income).astype('int64'))
                    output_list.append(np.array(label_marital).astype('int64'))
                    yield output_list
