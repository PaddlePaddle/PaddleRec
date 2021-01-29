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

from collections import defaultdict
from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.config = config
        self.file_list = file_list
        self.init()

    def init(self):
        all_field_id = [
            '101', '109_14', '110_14', '127_14', '150_14', '121', '122', '124',
            '125', '126', '127', '128', '129', '205', '206', '207', '210',
            '216', '508', '509', '702', '853', '301'
        ]
        self.all_field_id_dict = defaultdict(int)
        self.max_len = self.config.get("hyper_parameters.max_len", 3)
        for i, field_id in enumerate(all_field_id):
            self.all_field_id_dict[field_id] = [False, i]
        self.padding = 0

    def __iter__(self):
        full_lines = []
        self.data = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    features = l.strip().split(',')
                    ctr = int(features[1])
                    ctcvr = int(features[2])

                    output = [(field_id, [])
                              for field_id in self.all_field_id_dict]
                    output_list = []
                    for elem in features[4:]:
                        field_id, feat_id = elem.strip().split(':')
                        if field_id not in self.all_field_id_dict:
                            continue
                        self.all_field_id_dict[field_id][0] = True
                        index = self.all_field_id_dict[field_id][1]
                        output[index][1].append(int(feat_id))

                    for field_id in self.all_field_id_dict:
                        visited, index = self.all_field_id_dict[field_id]
                        self.all_field_id_dict[field_id][0] = False
                        if len(output[index][1]) > self.max_len:
                            output_list.append(
                                np.array(output[index][1][:self.max_len])
                                .astype('int64'))
                        else:
                            for ii in range(self.max_len - len(output[index][
                                    1])):
                                output[index][1].append(self.padding)
                            output_list.append(
                                np.array(output[index][1]).astype('int64'))
                    output_list.append(np.array([ctr]).astype('int64'))
                    output_list.append(np.array([ctcvr]).astype('int64'))
                    yield output_list
