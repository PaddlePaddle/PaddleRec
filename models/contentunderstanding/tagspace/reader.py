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

import sys

import numpy as np

from paddlerec.core.reader import ReaderBase


class Reader(ReaderBase):
    def init(self):
        pass

    def _process_line(self, l):
        tag_size = 4
        neg_size = 3
        l = l.strip().split(",")
        pos_index = int(l[0])
        pos_tag = []
        pos_tag.append(pos_index)
        text_raw = l[1].split()
        text = [int(w) for w in text_raw]
        neg_tag = []
        max_iter = 100
        now_iter = 0
        sum_n = 0
        while (sum_n < neg_size):
            now_iter += 1
            if now_iter > max_iter:
                print("error : only one class")
                sys.exit(0)
            rand_i = np.random.randint(0, tag_size)
            if rand_i != pos_index:
                neg_index = rand_i
                neg_tag.append(neg_index)
                sum_n += 1
        return text, pos_tag, neg_tag

    def generate_sample(self, line):
        def data_iter():
            text, pos_tag, neg_tag = self._process_line(line)
            if text is None:
                yield None
                return
            yield [('text', text), ('pos_tag', pos_tag), ('neg_tag', neg_tag)]

        return data_iter
