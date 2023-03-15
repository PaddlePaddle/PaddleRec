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

from ..base.utils.dataset_checker_utils import (
    persist_dataset_meta, build_res_dict, CheckFailedError,
    UnsupportedDatasetTypeError, DatasetFileNotFoundError)


@persist_dataset_meta
def check_dataset(dataset_dir, dataset_type):
    try:
        if dataset_type == 'Dataset':
            # Custom dataset
            dataset_dir = osp.abspath(dataset_dir)
            if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
                raise DatasetFileNotFoundError(file_path=dataset_dir)

            tags = ['train', 'val', 'test']
            sample_cnts = dict()
            sample_paths = defaultdict(list)
            im_sizes = defaultdict(Counter)
            num_classes = 0
            # We randomly sample `max_recorded_sample_cnts` from each subset
            max_recorded_sample_cnts = 50

            for tag in tags:
                file_list = osp.join(dataset_dir, f'{tag}.txt')
                if not osp.exists(file_list):
                    if tag in ('train', 'val'):
                        # train and val file lists must exist
                        raise DatasetFileNotFoundError(
                            file_path=file_list,
                            solution=f"Ensure that both `train.txt` and `val.txt` exist in `{dataset_dir}`."
                        )
                    else:
                        # tag == 'test'
                        continue
                else:
                    with open(file_list, 'r') as f:
                        all_lines = f.readlines()

                    # Each line corresponds to a sample
                    sample_cnts[tag] = len(all_lines)

                    for line in all_lines:
                        # Use space as delimiter
                        # TODO: Support more delimiters
                        delim = ' '
                        parts = line.rstrip().split(delim)
                        if tag in ('train', 'val'):
                            valid_num_parts_lst = [2]
                        else:
                            # tag == 'test'
                            valid_num_parts_lst = [1, 2]
                        if len(parts) not in valid_num_parts_lst:
                            raise CheckFailedError(
                                f"The number of delimiter-separated items in each row in {file_list} should be a number in {valid_num_parts_lst} (current delimiter is '{delim}')."
                            )

                        if len(parts) == 2:
                            img_path, lab_path = parts
                        else:
                            # len(parts) == 1
                            img_path = parts[0]
                            lab_path = None

                        if len(sample_paths[tag]) < max_recorded_sample_cnts:
                            # NOTE: According to PaddleRec rules, paths recorded in file list are
                            # the paths relative to dataset root dir.
                            # We populate the list with a dict
                            sample_paths[tag].append({
                                'img': img_path,
                                'lab': lab_path
                            })

                        img_path = osp.join(dataset_dir, img_path)
                        if lab_path is not None:
                            lab_path = osp.join(dataset_dir, lab_path)
                        if not osp.exists(img_path):
                            raise DatasetFileNotFoundError(file_path=img_path)
                        if lab_path is not None and not osp.exists(lab_path):
                            raise DatasetFileNotFoundError(file_path=lab_path)

                        img = Image.open(img_path)
                        im_shape = img.size
                        im_sizes[tag][tuple(im_shape)] += 1

                        if lab_path is not None:
                            lab = Image.open(lab_path)
                            lab_shape = lab.size
                            if lab.mode not in ('L', 'P'):
                                raise CheckFailedError(
                                    f"`{lab_path}` can not be recognized by PaddleRec.",
                                    solution="Ensure that all masks are stored in gray-scale or pseudo-color format."
                                )
                            if im_shape != lab_shape:
                                raise CheckFailedError(
                                    f"Image and mask have different shapes (`{img_path}` and `{lab_path}`)."
                                )
                            # We have to load mask to memory
                            lab = np.asarray(lab)
                            num_classes = max(num_classes, int(lab.max() + 1))

            meta = build_res_dict(True)

            meta['num_classes'] = num_classes

            meta['train.samples'] = sample_cnts['train']
            meta['train.im_sizes'] = im_sizes['train']
            meta['train.sample_paths'] = sample_paths['train']

            meta['val.samples'] = sample_cnts['val']
            meta['val.im_sizes'] = im_sizes['val']
            meta['val.sample_paths'] = sample_paths['val']

            # PaddleRec does not use test subset
            meta['test.samples'] = sample_cnts.get('test', None)
            meta['test.im_sizes'] = im_sizes.get('test', None)
            meta['test.sample_paths'] = sample_paths.get('test', None)
        else:
            raise UnsupportedDatasetTypeError(dataset_type=dataset_type)
    except CheckFailedError as e:
        return build_res_dict(False, err_type=type(e), err_msg=str(e))
    else:
        return meta
