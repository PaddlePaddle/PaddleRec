# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import sys

file_path = sys.argv[1]
f = open(file_path, "r")
lines = f.readlines()
f.close()

sparse_slots = [
    "logid", "time", "userid", "gender", "age", "occupation", "movieid",
    "title", "genres", "label"
]
for line in lines:
    line = line.strip().split(" ")
    title = []
    genres = []
    for i in line:
        if i.strip().split(":")[0] == "title":
            title.append(i.strip().split(":")[1])
        if i.strip().split(":")[0] == "genres":
            genres.append(i.strip().split(":")[1])
    title = title[:4]
    genres = genres[:3]
    if len(title) <= 4:
        for i in range(4 - len(title)):
            title.append("0")
    if len(genres) <= 3:
        for i in range(3 - len(genres)):
            genres.append("0")
    print(
        line[0] + " " + line[1] + " " + line[2] + " " + line[3] + " " + line[4]
        + " " + line[5] + " " + line[6] + " ",
        end="")
    for i in title:
        print("title:" + i + " ", end="")
    for i in genres:
        print("genres:" + i + " ", end="")
    print(line[-1])
