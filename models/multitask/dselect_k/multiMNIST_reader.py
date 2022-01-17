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
import pickle
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
            with open(file, "rb") as rf:
                x, y = pickle.load(rf)
            n_x = len(x)
            x = x.reshape(n_x, 1, 36, 36)
            for feature, label in zip(x, y):
                output_list = [
                    np.array(feature).astype("float32") / 255.0,
                    np.array(label[0]).astype("int64"),
                    np.array(label[1]).astype("int64")
                ]
                yield output_list
