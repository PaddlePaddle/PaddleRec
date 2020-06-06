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

train = dict()
test = dict()
data_path = "ml-1m"

for line in open(data_path + "/ratings.dat"):
    fea = line.rstrip().split("::")
    if fea[0] not in train:
        train[fea[0]] = [line]
    elif fea[0] not in test:
        test[fea[0]] = dict()
        test[fea[0]]['time'] = int(fea[3])
        test[fea[0]]['content'] = line
    else:
        time = int(fea[3])
        if time <= test[fea[0]]['time']:
            train[fea[0]].append(line)
        else:
            train[fea[0]].append(test[fea[0]]['content'])
            test[fea[0]]['time'] = time
            test[fea[0]]['content'] = line

train_data = []
for key in train:
    for line in train[key]:
        train_data.append(line)

random.shuffle(train_data)

with open(data_path + "/train.dat", 'w') as f:
    for line in train_data:
        f.write(line)

with open(data_path + "/test.dat", 'w') as f:
    for key in test:
        f.write(test[key]['content'])
