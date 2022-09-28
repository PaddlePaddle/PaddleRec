# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Reader(fleet.MultiSlotDataGenerator):
    def init(self, config):
        self.slots = ['user_id', 'item_id', 'label']
        logger.info("pipe init success")

    def line_process(self, line):
        features = line.strip().split(',')
        user_input = [int(features[0])]
        item_input = [int(features[1])]
        label = [int(features[2])]
        output_list = [(i, []) for i in self.slots]
        output_list[0][1].extend(user_input)
        output_list[1][1].extend(item_input)
        output_list[2][1].extend(label)
        return output_list

    def generate_sample(self, line):
        r"Dataset Generator"

        def reader():
            output_dict = self.line_process(line)
            yield output_dict

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
