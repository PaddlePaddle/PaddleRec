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
import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from base.register import register_model_info, register_suite_info
from .model import RankModel
from .runner import RankRunner
from .config import RankConfig
from .check_dataset import check_dataset

# XXX: Hard-code relative path of repo root dir
_file_path = osp.realpath(__file__)
REPO_ROOT_PATH = osp.abspath(osp.join(osp.dirname(_file_path), '..', '..'))
register_suite_info({
    'suite_name': 'Rank',
    'model': RankModel,
    'runner': RankRunner,
    'config': RankConfig,
    'dataset_checker': check_dataset,
    'runner_root_path': REPO_ROOT_PATH
})

# WideDeep 
WIDE_DEEP_CFG_PATH = "models/rank/wide_deep/config_gpups.yaml"
register_model_info({
    'model_name': 'wide_deep',
    'suite': 'Rank',
    'config_path': WIDE_DEEP_CFG_PATH,
    'auto_compression_config_path': WIDE_DEEP_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})

# DNN 
DNN_CFG_PATH = "models/rank/dnn/config_gpubox.yaml"
register_model_info({
    'model_name': 'dnn',
    'suite': 'Rank',
    'config_path': DNN_CFG_PATH,
    'auto_compression_config_path': DNN_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})
