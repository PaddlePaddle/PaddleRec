# -*- coding=utf8 -*-
"""
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
"""

from __future__ import print_function

from paddlerec.core.reader import ReaderBase
from paddlerec.core.utils import envs
import collections


class Reader(ReaderBase):
    def init(self):
        self.id2nodeid_path = envs.get_global_env("id2nodeid_path", None,
                                                  "hyper_parameters.tree")
        self.item_nums = envs.get_global_env("hyper_parameters.item_nums", 69)
        self.load_bidid_leafid(self.id2nodeid_path)
        pass

    def load_bidid_leafid(self, bidid_leafid_path):
        """load bidid2leafid
        """
        self.bidid_leafid = {}
        self.bidid_nodeid = {}
        with open(self.id2nodeid_path, 'r') as f:
            for index, line in enumerate(f):
                line = line.strip().split('\t')
                self.bidid_leafid[line[0]] = line[2]
                self.bidid_nodeid[line[0]] = line[1]

    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """

        self._all_slots_dict = collections.OrderedDict()
        for slot in range(self.item_nums):
            self._all_slots_dict["item_{}".format(slot + 1)] = [False, slot]

        def reader():
            """
            This function needs to be implemented by the user, based on data format
            """
            data = (line.strip('\n')).split('|')
            features = data[2].split(";")
            item_label = ((features[0].split("@"))[1].split(":"))[0]

            output = [(slot, []) for slot in self._all_slots_dict]
            output += [("item_mask_{}".format(i + 1), [])
                       for i in range(self.item_nums)]
            output += [("item_label", [int(self.bidid_leafid[item_label])])]

            for elem in features[1:]:
                feasign, value = elem.split("@")
                self._all_slots_dict[feasign][0] = True
                index = self._all_slots_dict[feasign][1]
                output[index][1].append(int(self.bidid_nodeid[value]))

            padding = 0
            mask = 1
            for slot in self._all_slots_dict:
                visit, index = self._all_slots_dict[slot]
                if visit:
                    self._all_slots_dict[slot][0] = False
                    output[index + self.item_nums][1].append(mask)
                else:
                    output[index][1].append(padding)
                    output[index + self.item_nums][1].append(padding)

            yield output

        return reader
