#encoding=utf-8     
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
"""
docstring
"""

import os
import sys

if len(sys.argv) < 2:
    print("usage:python {} input".format(sys.argv[0]))
    sys.exit(-1)

fin = open(sys.argv[1])
pos_num = 0
neg_num = 0

score_list = []
label_list = []
last_query = "-1"

#0       12.786960       1
#0       -1.480890       0
cnt = 0
query_num = 0
pair_num = 0
equal_num = 0
for line in fin:
    cols = line.strip().split("\t")
    cnt += 1
    if cnt % 500000 == 0:
        print("cnt:{}".format(1.0 * pos_num / neg_num))
    if len(cols) != 3:
        continue

    cur_query = cols[0]
    if cur_query != last_query:
        query_num += 1
        for i in range(0, len(score_list)):
            for j in range(i + 1, len(score_list)):
                if label_list[i] == label_list[j]:
                    continue
                pair_num += 1
                if (score_list[i] - score_list[j]) * (
                        label_list[i] - label_list[j]) < 0:
                    neg_num += 1
                elif (score_list[i] - score_list[j]) * (
                        label_list[i] - label_list[j]) > 0:
                    pos_num += 1
                else:
                    equal_num += 1
        score_list = []
        label_list = []

    last_query = cur_query

    label = int(cols[2])

    score_list.append(round(float(cols[1]), 6))
    label_list.append(int(cols[2]))

fin.close()

for i in range(0, len(score_list)):
    for j in range(i + 1, len(score_list)):
        if label_list[i] == label_list[j]:
            continue
        pair_num += 1
        if (score_list[i] - score_list[j]) * (label_list[i] - label_list[j]
                                              ) < 0:
            neg_num += 1
        elif (score_list[i] - score_list[j]) * (label_list[i] - label_list[j]
                                                ) > 0:
            pos_num += 1
        else:
            equal_num += 1

if neg_num > 0:
    print("pnr:{}".format(1.0 * pos_num / neg_num))
    print("query_num:{}".format(query_num))
    print("pair_num:{} , {}".format(pos_num + neg_num + equal_num, pair_num))
    print("equal_num:{}".format(equal_num))
    print("PNR: {}".format(1.0 * pos_num / (pos_num + neg_num)))
print("pos_num: {} , neg_num: {}".format(pos_num, neg_num))
