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

try:
    import cPickle as pickle
except ImportError:
    import pickle

from paddlerec.core.reader import Reader


class TrainReader(Reader):
    def init(self):
        pass

    def _process_line(self, line):
        line = line.strip().split(',')
        features = list(map(float, line))
        wide_feat = features[0:8]
        deep_feat = features[8:58+8]
        label = features[-1]
        return wide_feat, deep_feat, [label]
    
    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """
        def data_iter():
            wide_feat, deep_deat, label = self._process_line(line)
            yield [('wide_input', wide_feat), ('deep_input', deep_deat), ('label', label)]

        return data_iter