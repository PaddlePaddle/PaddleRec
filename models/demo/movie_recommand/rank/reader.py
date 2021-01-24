#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list

    def __iter__(self):
        full_lines = []
        self.data = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    output_list = []
                    line = l.strip().split(" ")
                    sparse_slots = [
                        "logid", "time", "userid", "gender", "age",
                        "occupation", "movieid", "title", "genres", "label"
                    ]
                    logid = line[0].strip().split(":")[1]
                    time = line[1].strip().split(":")[1]

                    userid = line[2].strip().split(":")[1]
                    output_list.append(np.array([int(userid)]))

                    gender = line[3].strip().split(":")[1]
                    output_list.append(np.array([int(gender)]))

                    age = line[4].strip().split(":")[1]
                    output_list.append(np.array([int(age)]))

                    occupation = line[5].strip().split(":")[1]
                    output_list.append(np.array([int(occupation)]))

                    movieid = line[6].strip().split(":")[1]
                    output_list.append(np.array([int(movieid)]))

                    title = []
                    genres = []
                    for i in line:
                        if i.strip().split(":")[0] == "title":
                            title.append(int(i.strip().split(":")[1]))
                        if i.strip().split(":")[0] == "genres":
                            genres.append(int(i.strip().split(":")[1]))
                    output_list.append(np.array(title))
                    output_list.append(np.array(genres))

                    label = line[-1].strip().split(":")[1]
                    output_list.append(np.array([int(label)]))

                    yield output_list
