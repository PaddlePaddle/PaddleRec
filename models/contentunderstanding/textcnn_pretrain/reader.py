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

from paddlerec.core.reader import ReaderBase


class Reader(ReaderBase):
    def init(self):
        pass

    def _process_line(self, l):
        l = l.strip().split()
        data = l[0:100]
        seq_len = l[100:101]
        label = l[101:]

        return data, label, seq_len

    def generate_sample(self, line):
        def data_iter():
            data, label, seq_len = self._process_line(line)
            if data is None:
                yield None
                return
            data = [int(i) for i in data]
            label = [int(i) for i in label]
            seq_len = [int(i) for i in seq_len]
            yield [('data', data), ('seq_len', seq_len), ('label', label)]

        return data_iter
