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
import paddle.fluid.incubate.data_generator as dg
import math
import os

try:
    import cPickle as pickle
except ImportError:
    import pickle


class Reader(dg.MultiSlotDataGenerator):
    def __init__(self, config):
        dg.MultiSlotDataGenerator.__init__(self)

    def init(self):
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [
            5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 11,
            231, 4008, 7393
        ]
        self.cont_diff_ = [
            self.cont_max_[i] - self.cont_min_[i]
            for i in range(len(self.cont_min_))
        ]
        self.cont_idx_ = list(range(1, 14))
        self.cat_idx_ = list(range(14, 40))

        dense_feat_names = ['I' + str(i) for i in range(1, 14)]
        sparse_feat_names = ['C' + str(i) for i in range(1, 27)]
        target = ['label']

        self.label_feat_names = target + dense_feat_names + sparse_feat_names

        self.cat_feat_idx_dict_list = [{} for _ in range(26)]

        # TODO: set vocabulary dictionary
        vocab_dir = "./sample_data/vocab/"
        for i in range(26):
            lookup_idx = 1  # remain 0 for default value
            for line in open(
                    os.path.join(vocab_dir, 'C' + str(i + 1) + '.txt')):
                self.cat_feat_idx_dict_list[i][line.strip()] = lookup_idx
                lookup_idx += 1

    def _process_line(self, line):
        features = line.rstrip('\n').split('\t')
        label_feat_list = [[] for _ in range(40)]
        for idx in self.cont_idx_:
            if features[idx] == '':
                label_feat_list[idx].append(0)
            else:
                # 0-1 minmax norm
                # label_feat_list[idx].append((float(features[idx]) - self.cont_min_[idx - 1]) /
                #                             self.cont_diff_[idx - 1])
                # log transform
                label_feat_list[idx].append(
                    math.log(4 + float(features[idx]))
                    if idx == 2 else math.log(1 + float(features[idx])))
        for idx in self.cat_idx_:
            if features[idx] == '' or features[
                    idx] not in self.cat_feat_idx_dict_list[idx - 14]:
                label_feat_list[idx].append(0)
            else:
                label_feat_list[idx].append(self.cat_feat_idx_dict_list[
                    idx - 14][features[idx]])
        label_feat_list[0].append(int(features[0]))
        return label_feat_list

    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """

        def data_iter():
            label_feat_list = self._process_line(line)
            s = ""
            for i in list(zip(self.label_feat_names, label_feat_list)):
                k = i[0]
                v = i[1]
                for j in v:
                    s += " " + k + ":" + str(j)
            print s.strip()
            yield None

        return data_iter


reader = Reader("../config.yaml")
reader.init()
reader.run_from_stdin()
