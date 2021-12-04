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
import pickle


class RecDataset(IterableDataset):
    def __init__(self, data_path, config):
        super(RecDataset, self).__init__()
        self.data_dir = data_path
        self.config = config
        self.batch_size = self.config.get("runner.data_batch_size")
        self.max_len = self.config.get(
            "hyper_parameters._max_position_seq_len")

    def __iter__(self):
        cnt = 0
        for file in self.data_dir:
            cnt += 1
        # Train: only use train data
        if cnt == 1:
            file = self.data_dir[0]
            with open(file, "r") as rf:
                sample_count = 0
                for line in rf:
                    if sample_count % self.batch_size == 0:
                        src_ids = []
                        pos_ids = []
                        input_mask = []
                        mask_pos = []
                        mask_label = []
                    output_list = []
                    split_samples = line.split(";")
                    tmp_ids = split_samples[1].split(',')
                    src_ids.append([int(x) for x in tmp_ids])
                    tmp_pos = split_samples[3].split(',')
                    pos_ids.append([int(x) for x in tmp_pos])
                    tmp_mask = split_samples[2].split(',')
                    input_mask.append([[int(x)] for x in tmp_mask])
                    tmp_mask_pos = split_samples[4].split(',')
                    mask_pos = mask_pos + [[
                        int(x) +
                        (sample_count % self.batch_size) * self.max_len
                    ] for x in tmp_mask_pos]
                    tmp_label = split_samples[5].split(',')
                    mask_label = mask_label + [[int(x)] for x in tmp_label]
                    sample_count += 1
                    if sample_count % self.batch_size == 0:
                        src_ids = np.array(src_ids)
                        pos_ids = np.array(pos_ids)
                        input_mask = np.array(input_mask)
                        mask_pos = np.array(mask_pos)
                        mask_label = np.array(mask_label)
                        output_list.append(src_ids)
                        output_list.append(pos_ids)
                        output_list.append(input_mask)
                        output_list.append(mask_pos)
                        output_list.append(mask_label)
                        yield output_list

        # Infer: use test data and candidate data
        else:
            test = None
            candidate_path = None
            for file in self.data_dir:
                if 'candidate' in file:
                    candidate_path = file
                else:
                    test = file
            with open(candidate_path, 'rb') as input_file:
                cand_list = pickle.load(input_file)

            with open(test, "r") as rf:
                sample_count = 0
                for line in rf:
                    if sample_count % self.batch_size == 0:
                        src_ids = []
                        pos_ids = []
                        input_mask = []
                        mask_pos = []
                        mask_label = []
                        candidate_ = []
                    output_list = []
                    split_samples = line.split(";")
                    tmp_ids = split_samples[1].split(',')
                    src_ids.append([int(x) for x in tmp_ids])
                    tmp_pos = split_samples[3].split(',')
                    pos_ids.append([int(x) for x in tmp_pos])
                    tmp_mask = split_samples[2].split(',')
                    input_mask.append([[int(x)] for x in tmp_mask])
                    tmp_mask_pos = split_samples[4].split(',')
                    mask_pos = mask_pos + [[
                        int(x) +
                        (sample_count % self.batch_size) * self.max_len
                    ] for x in tmp_mask_pos]
                    tmp_label = split_samples[5].split(',')
                    mask_label = mask_label + [[int(x)] for x in tmp_label]

                    tmp_candidate = cand_list[sample_count % self.batch_size]
                    candidate_.append(tmp_candidate)

                    sample_count += 1
                    if sample_count % self.batch_size == 0:
                        src_ids = np.array(src_ids)
                        pos_ids = np.array(pos_ids)
                        input_mask = np.array(input_mask)
                        mask_pos = np.array(mask_pos)
                        mask_label = np.array(mask_label)
                        output_list.append(src_ids)
                        output_list.append(pos_ids)
                        output_list.append(input_mask)
                        output_list.append(mask_pos)
                        output_list.append(mask_label)
                        candidate_ = np.array(candidate_)
                        output_list.append(candidate_)

                        cand_list = cand_list[self.batch_size:]
                        yield output_list
