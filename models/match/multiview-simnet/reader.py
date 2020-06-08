# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddlerec.core.reader import ReaderBase
from paddlerec.core.utils import envs


class Reader(ReaderBase):
    def init(self):
        self.query_slots = envs.get_global_env("hyper_parameters.query_slots",
                                               None, "train.model")
        self.title_slots = envs.get_global_env("hyper_parameters.title_slots",
                                               None, "train.model")

        self.all_slots = []
        for i in range(self.query_slots):
            self.all_slots.append(str(i))

        for i in range(self.title_slots):
            self.all_slots.append(str(i + self.query_slots))

        for i in range(self.title_slots):
            self.all_slots.append(str(i + self.query_slots + self.title_slots))

        self._all_slots_dict = dict()
        for index, slot in enumerate(self.all_slots):
            self._all_slots_dict[slot] = [False, index]

    def generate_sample(self, line):
        def data_iter():
            elements = line.rstrip().split()
            padding = 0
            output = [(slot, []) for slot in self.all_slots]
            for elem in elements:
                feasign, slot = elem.split(':')
                if not self._all_slots_dict.has_key(slot):
                    continue
                self._all_slots_dict[slot][0] = True
                index = self._all_slots_dict[slot][1]
                output[index][1].append(int(feasign))
            for slot in self._all_slots_dict:
                visit, index = self._all_slots_dict[slot]
                if visit:
                    self._all_slots_dict[slot][0] = False
                else:
                    output[index][1].append(padding)
            yield output

        return data_iter
