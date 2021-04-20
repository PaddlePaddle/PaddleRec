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
import random

class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.init()
        self.use_multi_task_learning = config.get("hyper_parameters.use_multi_task_learning")
        self.item_count = config.get("hyper_parameters.item_count")


    def init(self):
        pass

    def __iter__(self):
        full_lines = []
        self.data = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    line = l.strip().split()
                    item_id = int(line[-1])
                    user_embedding = []
                    for i in line[:-1]:
                        user_embedding.append(float(i))
                    output_list = []
                    output_list.append(
                        np.array(user_embedding).astype('float32'))
                    label = [item_id]
                    if self.use_multi_task_learning:
                            #label.append(random.randint(0,self.item_count -1))
                            label.append((item_id + self.item_count/2 ) % self.item_count)
                    output_list.append(
                        np.array(label).astype('int'))

                    yield output_list
