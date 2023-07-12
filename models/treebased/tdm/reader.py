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
import io
import six
import random
import os
import time
import sys
import paddle.distributed.fleet as fleet
import logging
#from paddle.distributed.fleet.dataset import TreeIndex

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class MyDataset(fleet.MultiSlotDataGenerator):
    def init(self, config):
        self.item_nums = config.get("hyper_parameters.item_nums", 69)
        self.sample_layer_counts = config.get(
            "hyper_parameters.tdm_layer_counts")
        self.tree_name = config.get("hyper_parameters.tree_name")

        self.start_sample_layer = config.get(
            "hyper_parameters.start_sample_layer", 1)
        self.with_hierachy = config.get("hyper_parameters.with_hierachy", True)
        self.seed = config.get("hyper_parameters.seed", 0)
        '''
        self.tree = TreeIndex(
            config.get("hyper_parameters.tree_name"),
            config.get("hyper_parameters.tree_path"))
        '''
        self.get_id_code('../builder/ids_id.txt')

    def get_id_code(self, _path):
        self.id_code = {}
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                key = line[0]
                self.id_code[key] = int(line[1])

    def line_process(self, line):
        #history_ids = [0] * (self.item_nums)
        history_ids = []
        features = line.strip().split(' ')
        item_id = self.id_code[features[-1]]
        for item in features[0:-1]:
            history_ids.append(float(item))
        #positive data 
        label = 1
        res = []
        res.append(history_ids)
        res.append(item_id)
        res.append(label)

        return res

    def generate_sample(self, line):
        "Dataset Generator"

        def reader():
            _ = self.line_process(line)
            feature_name = []
            #for i in range(self.item_nums):
            #    feature_name.append("item_" + str(i + 1))
            feature_name.append("user_item")
            feature_name.append("unit_id")
            feature_name.append("label")
            #output_list.reverse()
            output = []
            #tmp = _[:-2]
            output.append(_[0])
            output.append([int(_[-2])])
            output.append([int(_[-1])])
            yield zip(feature_name, output)

        return reader


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    utils_path = sys.argv[2]
    sys.path.append(utils_path)
    import common_ps
    yaml_helper = common_ps.YamlHelper()
    config = yaml_helper.load_yaml(yaml_path)

    r = MyDataset()
    r.init(config)
    r.run_from_stdin()
