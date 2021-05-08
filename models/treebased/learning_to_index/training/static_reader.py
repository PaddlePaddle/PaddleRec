# -*- coding=utf8 -*-
"""
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""

from __future__ import print_function
import sys
import os
import yaml
import paddle.distributed.fleet as fleet


class Reader(fleet.MultiSlotDataGenerator):
    def init(self, config):
        self.config = config

    def line_process(self, line):
        features = (line.strip('\n')).split('\t')
        input_emb = features[0].split(' ')
        input_emb = [float(i) for i in input_emb]
        item_label = [int(features[1])]
        return [input_emb] + [item_label]

    def generate_sample(self, line):
        "Dataset Generator"
        def reader():
            """
            This function needs to be implemented by the user, based on data format
            """
            inputs = self.line_process(line)
            feature_name = ["input_emb", "item_label"]
            yield list(zip(feature_name, inputs))

        return reader

    def dataloader(self, file_list):
        "DataLoader Pyreader Generator"
        def reader():
            for file in file_list:
                with open(file, 'r') as f:
                    for line in f:
                        input_data = self.line_process(line)
                        yield input_data

        return reader


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    utils_path = sys.argv[2]
    sys.path.append(utils_path)
    import common
    yaml_helper = common.YamlHelper()
    config = yaml_helper.load_yaml(yaml_path)

    r = Reader()
    r.init(config)
    r.run_from_stdin()
