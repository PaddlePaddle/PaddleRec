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
import sys

params = []
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip().strip('data: ').strip(',').split(',')
        line = map(float, line)
        params.append(line)

feas = []
with open(sys.argv[2]) as f:
    for line in f:
        line = line.strip().split('\t')
        feas.append(line)

score = []
with open(sys.argv[3]) as f:
    for line in f:
        line = float(line.strip().strip('data: ').strip()[1:-1])
        score.append(line)

assert (len(params) == len(feas))
length = len(params)

bias = None
for i in range(length):
    label = feas[i][-1]
    tmp = feas[i][2:-3]
    tmp_fea = feas[i][-3].split(":")
    _ = tmp_fea[1].split(" ")
    for j in range(len(_)):
        if _[j] != "":
            tmp.append(tmp_fea[0] + ":" + _[j])
    tmp_fea = feas[i][-2].split(":")
    _ = tmp_fea[1].split(" ")
    for j in range(len(_)):
        if _[j] != "":
            tmp.append(tmp_fea[0] + ":" + _[j])
    sort_p = np.argsort(np.array(params[i]))[::-1]

    res = []
    for j in range(len(sort_p)):
        res.append(tmp[sort_p[j]] + "_" + str(params[i][sort_p[j]]))

    res.append(label)
    res.append(str(score[i]))
    bias = score[i] - sum(params[i])
    print("; ".join(res))
    assert (len(params[i]) == len(tmp))
