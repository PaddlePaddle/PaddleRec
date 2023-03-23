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

import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from functools import lru_cache
from urllib.parse import urlparse

import yaml

from base import BaseConfig
from base.utils import logging
from base.utils.misc import abspath
from base.config_utils import merge_config


class RankConfig(BaseConfig):
    def update(self, dict_like_obj):
        dict_ = merge_config(self.dict, dict_like_obj)
        self.reset_from_dict(dict_)

    def load(self, config_path):
        pass

    def dump(self, config_path):
        pass

    def update_dataset(self, dataset_dir, dataset_type=None):
        dataset_dir = abspath(dataset_dir)
        if dataset_type is None:
            dataset_type = 'Dataset'
        if dataset_type == 'Dataset':
            ds_cfg = self._make_custom_dataset_config(dataset_dir)
            # For custom datasets, we reset the existing dataset configs
            self.update(ds_cfg)
        elif dataset_type == '_dummy':
            # XXX: A special dataset type to tease PaddleRec val dataset checkers
            self.update({
                'val_dataset': {
                    'type': 'Dataset',
                    'dataset_root': dataset_dir,
                    'val_path': os.path.join(dataset_dir, 'val.txt'),
                    'mode': 'val'
                },
            })
        else:
            raise ValueError(f"{dataset_type} is not supported.")

    def update_learning_rate(self, learning_rate):
        if 'lr_scheduler' not in self:
            raise RuntimeError(
                "Not able to update learning rate, because no LR scheduler config was found."
            )
        self.lr_scheduler['learning_rate'] = learning_rate

    def update_batch_size(self, batch_size, mode='train'):
        if mode == 'train':
            self.set_val('batch_size', batch_size)
        else:
            raise ValueError(
                f"Setting `batch_size` in {repr(mode)} mode is not supported.")

    def update_num_classes(self, num_classes):
        if 'train_dataset' in self:
            self.train_dataset['num_classes'] = num_classes
        if 'val_dataset' in self:
            self.val_dataset['num_classes'] = num_classes
        if 'model' in self:
            self.model['num_classes'] = num_classes

    def update_train_crop_size(self, crop_size):
        # XXX: This method is highly coupled to PaddleRec's internal functions
        if isinstance(crop_size, int):
            crop_size = [crop_size, crop_size]
        else:
            crop_size = list(crop_size)
            if len(crop_size) != 2:
                raise ValueError
            crop_size = [int(crop_size[0]), int(crop_size[1])]

        if 'train_dataset' in self and 'transforms' in self.train_dataset:
            tf_cfg_list = self.train_dataset['transforms']
            for tf_cfg in tf_cfg_list:
                if tf_cfg['type'] == 'RandomPaddingCrop':
                    tf_cfg['crop_size'] = crop_size
                    break
            else:
                logging.warn(
                    "Could not find configuration item of image cropping transformation operator. "
                    "Therefore, the crop size was not updated.")
        else:
            raise RuntimeError(
                "Not able to update crop size used in model training, "
                "because no training dataset config was found or the training dataset config does not contain transform configs."
            )

    def update_pretrained_weights(self, weight_path, is_backbone=False):
        if 'model' not in self:
            raise RuntimeError(
                "Not able to update pretrained weight path, because no model config was found."
            )
        if urlparse(weight_path).scheme == '':
            # If `weight_path` is not URL (with scheme present), 
            # it will be recognized as a local file path.
            weight_path = abspath(weight_path)
        if is_backbone:
            if 'backbone' not in self.model:
                raise RuntimeError(
                    "Not able to update pretrained weight path of backbone, because no backbone config was found."
                )
            self.model['backbone']['pretrained'] = weight_path
        else:
            self.model['pretrained'] = weight_path

    def _get_epochs_iters(self):
        if 'iters' in self:
            return self.iters
        else:
            # Default iters
            return 1000

    def _get_learning_rate(self):
        if 'lr_scheduler' not in self or 'learning_rate' not in self.lr_scheduler:
            # Default lr
            return 0.0001
        else:
            return self.lr_scheduler['learning_rate']

    def _get_batch_size(self, mode='train'):
        if 'batch_size' in self:
            return self.batch_size
        else:
            # Default batch size
            return 4

    def _get_qat_epochs_iters(self):
        return self.get_epochs_iters() // 2

    def _get_qat_learning_rate(self):
        return self.get_learning_rate() / 2

    def _update_dy2st(self, dy2st):
        self.set_val('to_static_training', dy2st)

    def _make_custom_dataset_config(self, dataset_root_path):
        # TODO: Description of dataset protocol
        ds_cfg = {
            'train_dataset': {
                'type': 'Dataset',
                'dataset_root': dataset_root_path,
                'train_path': os.path.join(dataset_root_path, 'train.txt'),
                'mode': 'train'
            },
            'val_dataset': {
                'type': 'Dataset',
                'dataset_root': dataset_root_path,
                'val_path': os.path.join(dataset_root_path, 'val.txt'),
                'mode': 'val'
            },
        }

        return ds_cfg

    # TODO: A full scanning of dataset can be time-consuming. 
    # Maybe we should cache the result to disk to avoid rescanning in another run?
    @lru_cache(8)
    def _extract_dataset_metadata(self, dataset_root_path, dataset_type):
        from .check_dataset import check_dataset
        meta = check_dataset(dataset_root_path, dataset_type)
        if not meta:
            # Return an empty dict
            return dict()
        else:
            return meta

    def __repr__(self):
        pass
