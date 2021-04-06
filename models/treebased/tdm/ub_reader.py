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
from paddle.distributed.fleet.data_generator import TreeIndex

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

        self.set_tree_layerwise_sampler(
            self.tree_name,
            self.sample_layer_counts,
            range(self.item_nums),
            self.item_nums,
            self.item_nums + 1,
            start_sample_layer=self.start_sample_layer,
            seed=self.seed,
            with_hierarchy=self.with_hierachy)

    def line_process(self, line):
        #378254_6|378254_6|train_unit_id@1045081:1.0;item_59@4856095;item_65@1821603;item_64@3598037;item_67@3423855;item_66@3598037;item_61@596274;item_60@3885113;item_63@3338392;item_62@643355;item_69@4951278;item_68@3308390||1.0|
        features = line.strip().split("|")[2].split(";")
        items = 0
        output_list = [[0]] * (self.item_nums + 2)
        for item in features:
            f = item.split("@")
            if f[0] == "train_unit_id":
                items = int(f[1].split(":")[0])
            else:
                output_list[int(f[0].split('_')[1]) - 1] = [int(f[1])]

        output_list[-2] = [items]
        output_list[-1] = [1]
        return output_list

    def generate_sample(self, line):
        "Dataset Generator"

        def reader():
            output_list = self.line_process(line)
            feature_name = []
            for i in range(self.item_nums):
                feature_name.append("item_" + str(i + 1))
            feature_name.append("unit_id")
            feature_name.append("label")
            yield zip(feature_name, output_list)

        return reader


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    utils_path = sys.argv[2]
    sys.path.append(utils_path)
    import common
    yaml_helper = common.YamlHelper()
    config = yaml_helper.load_yaml(yaml_path)

    r = MyDataset()
    tree = TreeIndex(
        config.get("hyper_parameters.tree_name"),
        config.get("hyper_parameters.tree_path"))
    r.init(config)
    r.run_from_stdin()
