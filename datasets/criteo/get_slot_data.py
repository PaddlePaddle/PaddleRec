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

import os
import paddle.fluid.incubate.data_generator as dg
try:
    import cPickle as pickle
except ImportError:
    import pickle

import paddle.fluid.incubate.data_generator as dg


class Reader(dg.MultiSlotDataGenerator):
    def __init__(self, config):
        dg.MultiSlotDataGenerator.__init__(self)

    def init(self):
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [
            5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 46,
            231, 4008, 7393
        ]
        self.cont_diff_ = [
            self.cont_max_[i] - self.cont_min_[i]
            for i in range(len(self.cont_min_))
        ]
        self.continuous_range_ = range(1, 14)
        self.categorical_range_ = range(14, 40)
        # load preprocessed feature dict	
        self.feat_dict_name = "sample_data/feat_dict_10.pkl2"
        self.feat_dict_ = pickle.load(open(self.feat_dict_name, 'rb'))

    def _process_line(self, line):
        features = line.rstrip('\n').split('\t')
        feat_idx = []
        feat_value = []
        for idx in self.continuous_range_:
            if features[idx] == '':
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(self.feat_dict_[idx])
                feat_value.append(
                    (float(features[idx]) - self.cont_min_[idx - 1]) /
                    self.cont_diff_[idx - 1])
        for idx in self.categorical_range_:
            if features[idx] == '' or features[idx] not in self.feat_dict_:
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(self.feat_dict_[features[idx]])
                feat_value.append(1.0)
        label = [int(features[0])]
        return feat_idx, feat_value, label

    def generate_sample(self, line):
        """	
        Read the data line by line and process it as a dictionary	
        """

        def data_iter():
            feat_idx, feat_value, label = self._process_line(line)
            s = ""
            for i in [('feat_idx', feat_idx), ('feat_value', feat_value),
                      ('label', label)]:
                k = i[0]
                v = i[1]
                for j in v:
                    s += " " + k + ":" + str(j)
            print(s.strip())  # add print for data preprocessing	
            yield None

        return data_iter


reader = Reader("../config.yaml")
reader.init()
reader.run_from_stdin()
