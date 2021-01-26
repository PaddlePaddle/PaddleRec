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

filename = './result.txt'
f = open(filename, "r")
lines = f.readlines()
f.close()
result = []
for line in lines:
    if "query_doc_sim" in str(line):
        result.append(line)
result = result[:-1]
f = open(filename, "w")
for i in range(len(result)):
    f.write(str(result[i]))
f.close()

label = []
filename = '../../../datasets/BQ_dssm/label.txt'
f = open(filename, "r")
#f.readline()
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
    line[3] = line[3].split(":")
    line = line[3][1].strip(" ")
    line = line.strip("[")
    line = line.strip("]")
    sim.append(float(line))

filename = '../../../datasets/BQ_dssm/big_test/test.txt'
f = open(filename, "r")
#f.readline()
query = []
for line in f.readlines():
    line = line.strip().split("\t")
    query.append(line[0])
f.close()


def takeFirst(x):
    return x[0]


filename = 'pair.txt'
line = []
print(len(query), len(sim), len(label))
for i in range(len(sim)):
    line.append([str(query[i]), str(sim[i]), str(label[i])])
line.sort(key=takeFirst)
f = open(filename, "w")
for i in line:
    f.write(i[0] + "\t" + i[1] + "\t" + i[2] + "\n")
f.close()
