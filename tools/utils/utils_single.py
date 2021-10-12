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


def reset_auc(use_fleet=False, auc_num=1):
    # for static clear auc
    auc_var_name = []
    for i in range(auc_num * 5):
        auc_var_name.append("_generated_var_{}".format(i))

    for name in auc_var_name:
        param = paddle.static.global_scope().find_var(name)
        if param == None:
            continue
        tensor = param.get_tensor()
        if param:
            tensor_array = np.zeros(tensor._get_dims()).astype("int64")
            if use_fleet:
                trainer_id = paddle.distributed.get_rank()
                tensor.set(tensor_array, paddle.CUDAPlace(trainer_id))
            else:
                tensor.set(tensor_array, paddle.CPUPlace())
            logger.info("AUC Reset To Zero: {}".format(name))


def auc(stat_pos, stat_neg, scope, util):
    stat_pos = np.array(scope.find_var(stat_pos.name).get_tensor())
    stat_neg = np.array(scope.find_var(stat_neg.name).get_tensor())

    # auc pos bucket shape
    old_pos_shape = np.array(stat_pos.shape)
    # reshape to one dim
    stat_pos = stat_pos.reshape(-1)
    global_pos = np.copy(stat_pos) * 0
    # reshape to its original shape

    global_pos = util.all_reduce(stat_pos, "sum")
    global_pos = global_pos.reshape(old_pos_shape)

    # auc neg bucket
    old_neg_shape = np.array(stat_neg.shape)
    stat_neg = stat_neg.reshape(-1)
    global_neg = np.copy(stat_neg) * 0

    global_neg = util.all_reduce(stat_neg, "sum")
    global_neg = global_neg.reshape(old_neg_shape)

    # calculate auc
    num_bucket = len(global_pos[0])
    area = 0.0
    pos = 0.0
    neg = 0.0
    new_pos = 0.0
    new_neg = 0.0
    total_ins_num = 0
    for i in range(num_bucket):
        index = num_bucket - 1 - i
        new_pos = pos + global_pos[0][index]
        total_ins_num += global_pos[0][index]
        new_neg = neg + global_neg[0][index]
        total_ins_num += global_neg[0][index]
        area += (new_neg - neg) * (pos + new_pos) / 2
        pos = new_pos
        neg = new_neg

    auc_value = None
    if pos * neg == 0 or total_ins_num == 0:
        auc_value = 0.5
    else:
        auc_value = area / (pos * neg)

    return auc_value
