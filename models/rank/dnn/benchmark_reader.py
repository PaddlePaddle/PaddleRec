# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import yaml
import six
import os
import copy
import paddle.distributed.fleet as fleet
import logging

cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Reader(fleet.MultiSlotDataGenerator):
    def init(self, config):
        self.config = config

    def line_process(self, line):
        features = line.rstrip('\n').split('\t')
        dense_feature = []
        sparse_feature = []
        for idx in continuous_range_:
            if features[idx] == "":
                dense_feature.append(0.0)
            else:
                dense_feature.append(
                    (float(features[idx]) - cont_min_[idx - 1]) /
                    cont_diff_[idx - 1])
        for idx in categorical_range_:
            sparse_feature.append([hash(str(idx) + features[idx]) % hash_dim_])
        label = [int(features[0])]
        return [label] + sparse_feature + [dense_feature]

    def generate_sample(self, line):
        "Dataset Generator"

        def reader():
            input_data = self.line_process(line)
            feature_name = ["dense_input"]
            for idx in categorical_range_:
                feature_name.append("C" + str(idx - 13))
            feature_name.append("label")
            yield zip(feature_name, input_data)

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
