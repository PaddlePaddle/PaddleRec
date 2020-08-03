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

from __future__ import print_function
import random
import pickle


def print_to_file(data, fout, slot):
    if not isinstance(data, list):
        data = [data]
    for i in range(len(data)):
        fout.write(slot + ":" + str(data[i]))
        fout.write(' ')


fake_seed_users = [i for i in range(2, 20)]
target_user = [1]
length = 100
print("make train data")
with open("paddle_train.txt", "w") as fout:
    for _ in range(length):

        print_to_file(fake_seed_users, fout, "user_seeds")
        print_to_file(target_user, fout, "target_user")
        print_to_file(1, fout, "label")
        fout.write("\n")
