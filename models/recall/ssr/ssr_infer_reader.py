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

from paddlerec.core.reader import Reader
from paddlerec.core.utils import envs


class EvaluateReader(Reader):
    def init(self):
        self.vocab_size = envs.get_global_env("vocab_size", 10, "train.model.hyper_parameters")

    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """

        def reader():
            """
            This function needs to be implemented by the user, based on data format
            """
            ids = line.strip().split()
            conv_ids = [int(i) for i in ids]
            boundary = len(ids) - 1
            src = conv_ids[:boundary]
            pos_tgt = [conv_ids[boundary]]
            feature_name = ["user", "all_item", "p_item"]
            yield zip(feature_name, [src] + [np.arange(self.vocab_size).astype("int64").tolist()] + [pos_tgt])

        return reader
