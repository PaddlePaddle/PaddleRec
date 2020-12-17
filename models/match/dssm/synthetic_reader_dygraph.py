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


class SyntheticDataset(IterableDataset):
    def __init__(self, file_list):
        super(SyntheticDataset, self).__init__()
        self.file_list = file_list

    def __iter__(self):
        full_lines = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for line in rf:
                    output_list = []
                    features = line.rstrip('\n').split('\t')
                    query = [
                        float(feature) for feature in features[0].split(',')
                    ]
                    output_list.append(np.array(query))
                    pos_doc = [
                        float(feature) for feature in features[1].split(',')
                    ]
                    output_list.append(np.array(pos_doc))

                    for i in range(len(features) - 2):
                        output_list.append(
                            np.array([
                                float(feature)
                                for feature in features[i + 2].split(',')
                            ]))
                    yield output_list
