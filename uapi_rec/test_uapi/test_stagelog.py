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

import os
import os.path as osp
import subprocess

import stagelog

from uapi_rec import PaddleModel, Config, check_dataset


def init_stagelog():
    os.environ['PADDLEX_TRACKING_SWITCH'] = 'TRUE'
    os.environ['PADDLEX_MODEL_ID'] = '1'
    os.environ['PADDLEX_TRACKING_URL'] = '{SENSITIVE_DATA}'

    run_id, token = stagelog.init()
    print(run_id, token)


def close_stagelog():
    stagelog.exit()


if __name__ == '__main__':

    model_name = 'wide_deep'

    save_dir = f"test_uapi/output/rec_res"
    dataset_dir = ""
    iters = 10
    init_stagelog()

    # Check dataset + success
    check_dataset(model_name, dataset_dir, dataset_type='Dataset')

    # Check dataset + failure
    check_dataset(model_name, dataset_dir, dataset_type='baidu')

    config = Config(model_name)
    config.update_dataset(dataset_dir)
    model = PaddleModel(model_name, config=config)

    # Train + success
    model.train(
        dataset=dataset_dir,
        batch_size=1,
        epochs_iters=iters,
        device='gpu',
        save_dir=save_dir)

    # Train + failure
    try:
        model.train(
            dataset=dataset_dir,
            epochs_iters=iters,
            device='baidu',
            save_dir=save_dir)
    except subprocess.CalledProcessError as e:
        print(str(e))

    close_stagelog()
