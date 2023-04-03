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

import os.path as osp
from collections import defaultdict, Counter

import numpy as np
from PIL import Image
import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from base.utils.dataset_checker_utils import (
    persist_dataset_meta, build_res_dict, CheckFailedError,
    UnsupportedDatasetTypeError, DatasetFileNotFoundError)

from base.utils import stagelog

MAX_V = 18446744073709551615


@persist_dataset_meta
def check_dataset(model_name, dataset_dir, dataset_type):
    stage_id = stagelog.running_datacheck(
        data_path=dataset_dir, data_type=dataset_type)
    try:
        if dataset_type == 'Dataset':
            # Custom dataset
            dataset_dir = osp.abspath(dataset_dir)
            if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
                raise DatasetFileNotFoundError(file_path=dataset_dir)

            tags = ['train']
            sample_cnts = dict()
            im_sizes = defaultdict(Counter)
            num_classes = 0
            # We randomly sample `max_recorded_sample_cnts` from each subset
            max_recorded_sample_cnts = 50

            for tag in tags:
                file_list = osp.join(dataset_dir, f'sample_{tag}.txt')
                if not osp.exists(file_list):
                    # train file lists must exist
                    raise DatasetFileNotFoundError(
                        file_path=file_list,
                        solution=f"Ensure that `train.txt` exist in `{dataset_dir}`."
                    )
                else:
                    with open(file_list, 'r') as f:
                        all_lines = f.readlines()

                    # Each line corresponds to a sample
                    sample_cnts[tag] = len(all_lines)

                    for line in all_lines:
                        # Use space as delimiter
                        # TODO: Support more delimiters
                        delim = ' '
                        parts = line.rstrip().split(delim)[0]
                        label = parts.split(":")[1]
                        label = int(label)
                        print("label is ", label)
                        print("part is ", parts)
                        if label not in (0, 1):
                            raise CheckFailedError(
                                f"The label of data in each row should be 0 or 1, but received {parts[0]}."
                            )

            meta = build_res_dict(True)

            meta['train.samples'] = sample_cnts['train']

        else:
            raise UnsupportedDatasetTypeError(dataset_type=dataset_type)
    except CheckFailedError as e:
        stagelog.fail(stage_id, str(e))
        return build_res_dict(False, err_type=type(e), err_msg=str(e))
    else:
        stagelog.success_datacheck(
            stage_id,
            train_dataset=meta['train.samples'],
            validation_dataset=None,
            test_dataset=None)
        #validation_dataset=meta['val.samples'],
        #test_dataset=meta['test.samples'] or 0)
        return meta
