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
import numpy as np

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

fea_dict = {}


class Reader(fleet.MultiSlotDataGenerator):
    def init(self, config):
        self.config = config
        padding = 0
        #sparse_slots = "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
        self.slots = self.slot_reader(slot_num=300)
        self.slot2index = {}
        self.visit = {}
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding
        logger.info("pipe init success")

    def slot_reader(self, slot_num=0, slot_file='./slot'):
        slots = []
        # slot is not 0, label=1, 
        if slot_num > 0:
            for i in range(slot_num + 2):
                if i == 0:
                    continue
                slots.append(str(i))
        else:
            with open(slot_file, "r") as rf:
                for line in rf.readlines():
                    slots.append(line.strip())
        return slots

    def line_process(self, line):
        line = line.strip().split(" ")
        output = [(i, []) for i in self.slots]
        for i in line:
            slot_feasign = i.split(":")
            if len(slot_feasign) < 2:
                print(i)
            slot = slot_feasign[1]
            if slot not in self.slots:
                continue
            feasign = int(slot_feasign[0])
            # if feasign not in fea_dict:
            #     fea_dict[feasign] = len(fea_dict)
            # output[self.slot2index[slot]][1].append(fea_dict[feasign])
            output[self.slot2index[slot]][1].append(feasign)
            self.visit[slot] = True
        for i in self.visit:
            slot = i
            if not self.visit[slot]:
                output[self.slot2index[i]][1].extend([self.padding])
            else:
                self.visit[slot] = False

        # add show
        output = [("0", [1])] + output
        return output

    def generate_sample(self, line):
        "Dataset Generator"

        def reader():
            output_dict = self.line_process(line)
            # {key, value} dict format: {'labels': [1], 'sparse_slot1': [2, 3], 'sparse_slot2': [4, 5, 6, 8], 'dense_slot': [1,2,3,4]} 
            # dict must match static_model.create_feed()
            yield output_dict

        return reader


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    utils_path = sys.argv[2]
    sys.path.append(utils_path)
    import common_ps
    yaml_helper = common_ps.YamlHelper()
    config = yaml_helper.load_yaml(yaml_path)

    r = Reader()
    r.init(config)
    r.run_from_stdin()
