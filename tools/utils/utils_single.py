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

from . import envs
import os
import copy
import subprocess
import sys
import argparse
import warnings
import logging
import paddle
import numpy as np
from paddle.io import DistributedBatchSampler, DataLoader

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def _mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--config_yaml", type=str)
    args = parser.parse_args()
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def get_abs_model(model):
    if model.startswith("paddlerec."):
        dir = envs.paddlerec_adapter(model)
        path = os.path.join(dir, "config.yaml")
    else:
        if not os.path.isfile(model):
            raise IOError("model config: {} invalid".format(model))
        path = model
    return path


def get_all_inters_from_yaml(file, filters):
    _envs = envs.load_yaml(file)
    all_flattens = {}

    def fatten_env_namespace(namespace_nests, local_envs):
        for k, v in local_envs.items():
            if isinstance(v, dict):
                nests = copy.deepcopy(namespace_nests)
                nests.append(k)
                fatten_env_namespace(nests, v)
            elif (k == "dataset" or k == "phase" or
                  k == "runner") and isinstance(v, list):
                for i in v:
                    if i.get("name") is None:
                        raise ValueError("name must be in dataset list. ", v)
                    nests = copy.deepcopy(namespace_nests)
                    nests.append(k)
                    nests.append(i["name"])
                    fatten_env_namespace(nests, i)
            else:
                global_k = ".".join(namespace_nests + [k])
                all_flattens[global_k] = v

    fatten_env_namespace([], _envs)
    ret = {}
    for k, v in all_flattens.items():
        for f in filters:
            if k.startswith(f):
                ret[k] = v
    return ret


def create_data_loader(config, place, mode="train"):
    if mode == "train":
        data_dir = config.get("runner.train_data_dir", None)
        batch_size = config.get('runner.train_batch_size', None)
        reader_path = config.get('runner.train_reader_path', 'reader')
    else:
        data_dir = config.get("runner.test_data_dir", None)
        batch_size = config.get('runner.infer_batch_size', None)
        reader_path = config.get('runner.infer_reader_path', 'reader')
    config_abs_dir = config.get("config_abs_dir", None)
    data_dir = os.path.join(config_abs_dir, data_dir)
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    user_define_reader = config.get('runner.user_define_reader', False)
    logger.info("reader path:{}".format(reader_path))
    from importlib import import_module
    reader_class = import_module(reader_path)
    dataset = reader_class.RecDataset(file_list, config=config)
    loader = DataLoader(
        dataset, batch_size=batch_size, places=place, drop_last=True)
    return loader


def load_dy_model_class(abs_dir):
    sys.path.append(abs_dir)
    from dygraph_model import DygraphModel
    dy_model = DygraphModel()
    return dy_model


def load_static_model_class(config):
    abs_dir = config['config_abs_dir']
    sys.path.append(abs_dir)
    from static_model import StaticModel
    static_model = StaticModel(config)
    return static_model


def load_yaml(yaml_file, other_part=None):
    part_list = ["workspace", "runner", "hyper_parameters"]
    if other_part:
        part_list += other_part
    running_config = get_all_inters_from_yaml(yaml_file, part_list)
    return running_config


def reset_auc():
    auc_var_name = [
        "_generated_var_0", "_generated_var_1", "_generated_var_2",
        "_generated_var_3"
    ]
    for name in auc_var_name:
        param = paddle.fluid.global_scope().var(name)
        if param == None:
            continue
        tensor = param.get_tensor()
        if param:
            tensor_array = np.zeros(tensor._get_dims()).astype("int64")
            tensor.set(tensor_array, paddle.CPUPlace())
            logger.info("AUC Reset To Zero: {}".format(name))
