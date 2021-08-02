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
import sklearn
import math
"""
Extracting information from infer data
"""
filename = './result.txt'
f = open(filename, "r")
lines = f.readlines()
f.close()
result = []
for line in lines:
    if "prediction" in str(line):
        result.append(line)
result = result[:-1]

pair = []
for line in result:
    line = line.strip().split(",")
    for seg in line:
        if "user" in seg:
            user_id = seg.strip().split(":")[1].strip(" ").strip("[]")
        if "prediction" in seg:
            prediction = seg.strip().split(":")[1].strip(" ").strip("[]")
        if "label" in seg:
            label = seg.strip().split(":")[1].strip(" ").strip("[]")
    pair.append([int(user_id), float(prediction), int(label)])


def takeSecond(x):
    return x[1]


"""
Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
"""
hits = []
ndcg = []
pair = [pair[i:i + 100] for i in range(0, len(pair), 100)]
for user in pair:
    user.sort(key=takeSecond, reverse=True)
    each_user_top10_line = user[:10]
    each_user_top10_line_label = [i[2] for i in each_user_top10_line]
    if 1 in each_user_top10_line_label:
        i = each_user_top10_line_label.index(1)
        ndcg.append(math.log(2) / math.log(i + 2))
        hits.append(1)
    else:
        hits.append(0)
        ndcg.append(0)

print("user_num:", len(hits))
print("hit ratio:", np.array(hits).mean())
print("ndcg:", np.array(ndcg).mean())
