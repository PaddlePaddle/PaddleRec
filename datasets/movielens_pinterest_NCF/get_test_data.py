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

import numpy as np

filename = './Data/ml-1m.test.negative'
f = open(filename, "r")
lines = f.readlines()
f.close()
filename = './test_data.csv'
f = open(filename, "w")
for line in lines:
    line = line.strip().split("\t")
    user_id = line[0].strip("()").split(",")[0]
    positive_item = line[0].strip("()").split(",")[1]
    negative_item = []
    for item in line[1:]:
        negative_item.append(int(item))

    f.write(user_id + "," + positive_item + "," + "1" + "\n")
    for item in negative_item:
        f.write(user_id + "," + str(item) + "," + "0" + "\n")

f.close()
