#!/usr/bin/env python

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import shutil
import os.path as osp
import sys, os
__dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append('../')
print(sys.path)
from rank.model import RankModel
from rank.config import RankConfig

if __name__ == '__main__':
    model_name = 'wide_deep'
    model_name = 'dnn'

    model = RankModel(model_name=model_name)

    # hyterparams will be set by config yaml in register.py
    save_dir = ""
    dataset_dir = ""
    if osp.exists(save_dir):
        shutil.rmtree(save_dir)

    print(model.supported_apis)

    # Do test
    model.train(
        dataset=dataset_dir,
        batch_size=1,
        epochs_iters=10,
        device='gpu:0,1',
        amp='O1',
        save_dir=save_dir)
