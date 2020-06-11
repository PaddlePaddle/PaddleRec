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

from paddlerec.core.reader import ReaderBase
from paddlerec.core.utils import envs
from collections import defaultdict
import numpy as np


class Reader(ReaderBase):
    def init(self):
        pass

    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """

        def reader():
            """
            This function needs to be implemented by the user, based on data format
            """
            features = line.strip().split(',')

            feature_name = ["user_input", "item_input"]
            yield list(
                zip(feature_name, [[int(features[0])]] + [[int(features[1])]]))

        return reader
