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

import os
import sys

from fleet_rec.core.utils.envs import lazy_instance
from fleet_rec.core.utils.envs import get_global_env


def dataloader(readerclass, train, yaml_file):
    namespace = "train.reader"

    if train == "TRAIN":
        reader_name = "TrainReader"
        data_path = get_global_env("train_data_path", None, namespace)
    else:
        reader_name = "EvaluateReader"
        data_path = get_global_env("test_data_path", None, namespace)

    files = [str(data_path) + "/%s" % x for x in os.listdir(data_path)]

    reader_class = lazy_instance(readerclass, reader_name)
    reader = reader_class(yaml_file)
    reader.init()

    def gen_reader():
        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    line = line.rstrip('\n')
                    iter = reader.generate_sample(line)
                    for parsed_line in iter():
                        if parsed_line is None:
                            continue
                        else:
                            values = []
                            for pased in parsed_line:
                                values.append(pased[1])
                            yield values
    return gen_reader
