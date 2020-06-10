# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import time
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default='sample',
    help='dataset dir: diginetica/yoochoose1_4/yoochoose1_64/sample')
opt = parser.parse_args()


def process_data(file_type):
    path = os.path.join(opt.data_dir, file_type)
    output_path = os.path.splitext(path)[0] + ".txt"
    data = pickle.load(open(path, 'rb'))
    data = list(zip(data[0], data[1]))
    length = len(data)
    with open(output_path, 'w') as fout:
        for i in range(length):
            fout.write(','.join(map(str, data[i][0])))
            fout.write('\t')
            fout.write(str(data[i][1]))
            fout.write("\n")


process_data("train")
process_data("test")

print('Done.')
