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
        if config:
            self.CE = config.get("runner.CE", False)

    def __iter__(self):
        full_lines = []
        self.data = []
        if self.CE:
            np.random.seed(12345)
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    output_list = []
                    tag_size = 4
                    neg_size = 3
                    text_size = 45
                    _pad_ = 75377
                    line = l.strip().split(",")
                    pos_index = int(line[0])
                    pos_tag = []
                    pos_tag.append(pos_index)
                    text_raw = line[1].split(" ")
                    text = [int(w) for w in text_raw]
                    if len(text) < text_size:
                        for i in range(text_size - len(text)):
                            text.append(_pad_)
                    else:
                        text = text[:text_size]

                    neg_tag = []
                    while (len(neg_tag) < neg_size):
                        rand_i = np.random.randint(0, tag_size)
                        if rand_i != pos_index:
                            neg_index = rand_i
                            neg_tag.append(neg_index)

                    output_list.append(np.array(text).astype('int64'))
                    output_list.append(np.array(pos_tag).astype('int64'))
                    output_list.append(np.array(neg_tag).astype('int64'))

                    yield output_list
