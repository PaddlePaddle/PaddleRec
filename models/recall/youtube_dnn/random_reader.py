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

from paddlerec.core.reader import ReaderBase
from paddlerec.core.utils import envs
from collections import defaultdict


class Reader(ReaderBase):
    def init(self):
        self.watch_vec_size = envs.get_global_env(
            "hyper_parameters.watch_vec_size")
        self.search_vec_size = envs.get_global_env(
            "hyper_parameters.search_vec_size")
        self.other_feat_size = envs.get_global_env(
            "hyper_parameters.other_feat_size")
        self.output_size = envs.get_global_env("hyper_parameters.output_size")

    def generate_sample(self, line):
        """
        the file is not used
        """

        def reader():
            """
            This function needs to be implemented by the user, based on data format
            """

            feature_name = ["watch_vec", "search_vec", "other_feat", "label"]
            yield list(
                zip(feature_name, [
                    np.random.rand(self.watch_vec_size).tolist()
                ] + [np.random.rand(self.search_vec_size).tolist()] + [
                    np.random.rand(self.other_feat_size).tolist()
                ] + [[np.random.randint(self.output_size)]]))

        return reader
