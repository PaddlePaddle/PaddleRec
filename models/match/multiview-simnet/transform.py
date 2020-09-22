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

import random
import numpy as np

label = []
filename = './data/label.txt'
f = open(filename, "r")
f.readline()
num = 0
for line in f.readlines():
    num = num + 1
    line = line.strip()
    label.append(line)
f.close()
print(num)

filename = './result.txt'
sim = []
for line in open(filename):
    line = line.strip().split(",")
    print(line)
    line[3] = line[3].split(":")
    line = line[3][1].strip(" ")
    line = line.strip("[")
    line = line.strip("]")
    sim.append(float(line))

filename = './data/testquery.txt'
f = open(filename, "r")
f.readline()
query = []
for line in f.readlines():
    line = line.strip()
    query.append(line)
f.close()

filename = 'pair.txt'
f = open(filename, "w")
for i in range(len(sim)):
    print(i)
    f.write(str(query[i]) + "\t" + str(sim[i]) + "\t" + str(label[i]) + "\n")
f.close()
