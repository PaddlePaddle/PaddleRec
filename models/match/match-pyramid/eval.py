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


def eval_MAP(pred, gt):
    map_value = 0.0
    r = 0.0
    c = list(zip(pred, gt))
    random.shuffle(c)
    c = sorted(c, key=lambda x: x[0], reverse=True)
    for j, (p, g) in enumerate(c):
        if g != 0:
            r += 1
            map_value += r / (j + 1.0)
    if r == 0:
        return 0.0
    else:
        return map_value / r


filename = './result.txt'
f = open(filename, "r")
lines = f.readlines()
f.close()
result = []
for line in lines:
    if "prediction" in str(line):
        result.append(line)
result = result[:-1]
f = open(filename, "w")
for i in range(len(result)):
    f.write(str(result[i]))
f.close()

filename = '../../../datasets/letor07/data/relation.test.fold1.txt'
gt = []
qid = []
f = open(filename, "r")
#f.readline()
num = 0
for line in f.readlines():
    num = num + 1
    line = line.strip().split()
    gt.append(int(line[0]))
    qid.append(line[1])
f.close()
print(num)
filename = './result.txt'
pred = []
for line in open(filename):
    line = line.strip().split(",")
    line[3] = line[3].split(":")
    line = line[3][1].strip(" ")
    line = line.strip("[")
    line = line.strip("]")
    pred.append(float(line))

result_dict = {}
print(len(pred))
print(len(qid))
for i in range(len(pred)):
    if qid[i] not in result_dict:
        result_dict[qid[i]] = []
    result_dict[qid[i]].append([gt[i], pred[i]])
print(len(result_dict))

map = 0
for qid in result_dict:
    gt = np.array(result_dict[qid])[:, 0]
    pred = np.array(result_dict[qid])[:, 1]
    map += eval_MAP(pred, gt)
map = map / len(result_dict)

print("map=", map)
