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
import sklearn.metrics

label = []
filename = './data/label.txt'
f = open(filename, "r")
f.readline()
num = 0
for line in f.readlines():
    num = num + 1
    line = line.strip()
    label.append(float(line))
f.close()
print(num)

filename = './result.txt'
sim = []
for line in open(filename):
    line = line.strip().split(",")
    line[1] = line[1].split(":")
    line = line[1][1].strip(" ")
    line = line.strip("[")
    line = line.strip("]")
    sim.append(float(line))

auc = sklearn.metrics.roc_auc_score(label, sim)
print("auc = ", auc)
