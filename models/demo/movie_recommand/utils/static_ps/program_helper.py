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
import common_ps
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
    assert sync_mode in ["async", "sync", "geo", "heter"]
    if sync_mode == "sync":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = False
    elif sync_mode == "async":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
    elif sync_mode == "geo":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"k_steps": config.get("runner.geo_step")}
    elif sync_mode == "heter":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"heter_worker_device_guard": "gpu"}
    return strategy


def get_model(config):
    abs_dir = config['config_abs_dir']
    sys.path.append(abs_dir)
    from static_model import StaticModel
    static_model = StaticModel(config)
    return static_model
