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

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Reader(fleet.MultiSlotStringDataGenerator):
    def init(self, config):
        self.config = config
        self.neg_num = self.config.get("hyper_parameters.neg_num")

    def line_process(self, line):
        data = line.rstrip('\n').split('\t')
        ins_id = [data[0]]
        content = [data[1]]
        features = data[2:]
        query = features[0].split(',')
        pos_doc = features[1].split(',')

        neg_doc_list = []
        for i in range(self.neg_num):
            neg_doc_list.append(features[i + 2].split(','))

        return [ins_id, content, query, pos_doc] + neg_doc_list

    def generate_sample(self, line):
        "Dataset Generator"

        def reader():
            input_data = self.line_process(line)
            feature_name = ["insid", "content", "query", "pos_doc"]
            for i in range(self.neg_num):
                feature_name.append("neg_doc_{}".format(i))
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
    import common_ps
    yaml_helper = common_ps.YamlHelper()
    config = yaml_helper.load_yaml(yaml_path)

    r = Reader()
    r.init(config)
    # r.init(None)
    r.run_from_stdin()
