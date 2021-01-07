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


class BQDataset(IterableDataset):
    def __init__(self, file_list):
        super(BQDataset, self).__init__()
        self.file_list = file_list

    def __iter__(self):
        full_lines = []
        self.data = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    output_list = []
                    line = l.strip().split(" ")

                    slot_0 = []
                    slot_1 = []
                    slot_2 = []
                    for i in line:
                        if i.strip().split(":")[0] == "0":
                            slot_0.append(float(i.strip().split(":")[1]))
                        if i.strip().split(":")[0] == "1":
                            slot_1.append(float(i.strip().split(":")[1]))
                        if i.strip().split(":")[0] == "2":
                            slot_2.append(float(i.strip().split(":")[1]))
                    output_list.append(np.array(slot_0))
                    output_list.append(np.array(slot_1))
                    output_list.append(np.array(slot_2))

                    yield output_list
