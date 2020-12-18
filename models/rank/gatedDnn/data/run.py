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

import os
import glob
import platform

os.system("sh download.sh")

os.mkdir("slot_train_data_full")
file_name = []
file_name = os.listdir("train_data_full")
sysstr = platform.system()
if (sysstr == "Linux"):
    for i in file_name:
        os.system(
            "cat train_data_full/{} | python get_slot_data.py > slot_train_data_full/{}".
            format(i, i))
if (sysstr == "Windows"):
    for i in file_name:
        os.system(
            "type train_data_full\{} | python get_slot_data.py > slot_train_data_full\{}".
            format(i, i))

os.mkdir("slot_test_data_full")
file_name = []
file_name = os.listdir("test_data_full")
sysstr = platform.system()
if (sysstr == "Linux"):
    for i in file_name:
        os.system(
            "cat test_data_full/{} | python get_slot_data.py > slot_test_data_full/{}".
            format(i, i))
if (sysstr == "Windows"):
    for i in file_name:
        os.system(
            "type test_data_full\{} | python get_slot_data.py > slot_test_data_full\{}".
            format(i, i))

os.mkdir("slot_train_data")
file_name = []
file_name = os.listdir("train_data")
sysstr = platform.system()
if (sysstr == "Linux"):
    for i in file_name:
        os.system(
            "cat train_data/{} | python get_slot_data.py > slot_train_data/{}".
            format(i, i))
if (sysstr == "Windows"):
    for i in file_name:
        os.system(
            "type train_data\{} | python get_slot_data.py > slot_train_data\{}".
            format(i, i))

os.mkdir("slot_test_data")
file_name = []
file_name = os.listdir("test_data")
sysstr = platform.system()
if (sysstr == "Linux"):
    for i in file_name:
        os.system(
            "cat test_data/{} | python get_slot_data.py > slot_test_data/{}".
            format(i, i))
if (sysstr == "Windows"):
    for i in file_name:
        os.system(
            "type test_data\{} | python get_slot_data.py > slot_test_data\{}".
            format(i, i))
