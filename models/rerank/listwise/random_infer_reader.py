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
import paddle.fluid as fluid

from paddlerec.core.reader import Reader
from paddlerec.core.utils import envs
from collections import defaultdict


class EvaluateReader(Reader):
    def init(self):
        self.user_vocab = envs.get_global_env("hyper_parameters.user_vocab",
                                              None, "train.model")
        self.item_vocab = envs.get_global_env("hyper_parameters.item_vocab",
                                              None, "train.model")
        self.item_len = envs.get_global_env("hyper_parameters.item_len", None,
                                            "train.model")
        self.batch_size = envs.get_global_env("batch_size", None,
                                              "train.reader")

    def reader_creator(self):
        def reader():
            user_slot_name = []
            for j in range(self.batch_size):
                user_slot_name.append(
                    [int(np.random.randint(self.user_vocab))])
            item_slot_name = np.random.randint(
                self.item_vocab, size=(self.batch_size,
                                       self.item_len)).tolist()
            length = [self.item_len] * self.batch_size
            label = np.random.randint(
                2, size=(self.batch_size, self.item_len)).tolist()
            output = []
            output.append(user_slot_name)
            output.append(item_slot_name)
            output.append(length)
            output.append(label)

            yield output

        return reader

    def generate_batch_from_trainfiles(self, files):
        return fluid.io.batch(
            self.reader_creator(), batch_size=self.batch_size)

    def generate_sample(self, line):
        """
        the file is not used
        """

        def reader():
            """
            This function needs to be implemented by the user, based on data format
            """
            pass

        return reader
