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

import os
import cv2
import numpy as np
import random
import shutil

data_folder = './omniglot_raw'  # omniglot数据集路径

character_folders = [os.path.join(data_folder, family, character) \
                     for family in os.listdir(data_folder) \
                     if os.path.isdir(os.path.join(data_folder, family)) \
                     for character in os.listdir(os.path.join(data_folder, family))]
print("The number of character folders: {}".format(len(
    character_folders)))  # 1623
random.seed(1)
random.shuffle(character_folders)
train_folders = character_folders[:973]
val_folders = character_folders[973:1298]
test_folders = character_folders[1298:]
print('The number of train characters is {}'.format(len(train_folders)))  # 973
print('The number of validation characters is {}'.format(len(
    val_folders)))  # 325
print('The number of test characters is {}'.format(len(test_folders)))  # 325

for char_fold in train_folders:
    path = char_fold.split("/")
    path[1] = "omniglot_train"
    shutil.copytree(char_fold, "/".join(path))

for char_fold in val_folders:
    path = char_fold.split("/")
    path[1] = "omniglot_valid"
    shutil.copytree(char_fold, "/".join(path))

for char_fold in test_folders:
    path = char_fold.split("/")
    path[1] = "omniglot_test"
    shutil.copytree(char_fold, "/".join(path))
