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

from __future__ import print_function
import os
import warnings
import logging
import paddle
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet
from . import common_ps
import sys

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_strategy(config):
    if not common_ps.is_distributed_env():
        logger.warn(
            "Not Find Distributed env, Change To local train mode. If you want train with fleet, please use [fleetrun] command."
        )
        return None
    sync_mode = config.get("runner.sync_mode")
    assert sync_mode in ["async", "sync", "geo", "heter", "gpubox"]
    if sync_mode == "sync":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = False
    elif sync_mode == "async":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
    elif sync_mode == "geo":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.is_with_coordinator = True if config.get(
            "runner.with_coordinator") == 1 else False
        a_sync_configs = strategy.a_sync_configs
        a_sync_configs["k_steps"] = config.get("runner.geo_step")
        strategy.a_sync_configs = a_sync_configs
    elif sync_mode == "heter":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"heter_worker_device_guard": "gpu"}
    elif sync_mode == "gpubox":
        print("sync_mode = {}".format(sync_mode))
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"use_ps_gpu": 1}

    strategy.trainer_desc_configs = {
        "dump_fields_path": config.get("runner.dump_fields_path", ""),
        "dump_fields": config.get("runner.dump_fields", []),
        "dump_param": config.get("runner.dump_param", []),
        "stat_var_names": config.get("stat_var_names", []),
        "local_sparse": config.get("runner.local_sparse", []),
        "remote_sparse": config.get("runner.remote_sparse", [])
    }
    print("strategy:", strategy.trainer_desc_configs)

    if config.get("runner.fs_client.uri") is not None:
        strategy.fs_client_param = {
            "uri": config.get("runner.fs_client.uri", ""),
            "user": config.get("runner.fs_client.user", ""),
            "passwd": config.get("runner.fs_client.passwd", ""),
            "hadoop_bin": config.get("runner.fs_client.hadoop_bin", "hadoop")
        }
    print("strategy:", strategy.fs_client_param)

    strategy.adam_d2sum = config.get("hyper_parameters.adam_d2sum", True)
    table_config = {}
    for x in config:
        if x.startswith("table_parameters"):
            table_name = x.split('.')[1]
            if table_name not in table_config:
                table_config[table_name] = {}
            table_config[table_name][x] = config[x]
    print("table_config:", table_config)
    strategy.sparse_table_configs = table_config
    print("strategy table config:", strategy.sparse_table_configs)

    return strategy


def get_model(config):
    abs_dir = config['config_abs_dir']
    sys.path.append(abs_dir)
    from static_model import StaticModel
    static_model = StaticModel(config)
    return static_model


def set_dump_config(program, dump_config):
    if dump_config.get("dump_fields_path") is not None:
        program._fleet_opt["dump_fields_path"] = dump_config.get(
            "dump_fields_path")
    if dump_config.get("dump_fields") is not None:
        program._fleet_opt["dump_fields"] = dump_config.get("dump_fields")
    if dump_config.get("dump_param") is not None:
        program._fleet_opt["dump_param"] = dump_config.get("dump_param")
