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
import sys
import warnings
import logging
import numpy as np
import time
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
import common

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_avg_cost_mins(value):
    t1 = time.time()
    local_cost = np.array([value])
    global_cost = np.copy(local_cost) * 0
    t2 = time.time()
    #fleet._role_maker._node_type_comm.Allreduce(local_cost, global_cost)
    global_cost = fleet.util.all_reduce(local_cost)
    t3 = time.time()
    avg_cost = float(global_cost[0]) / fleet.worker_num()
    avg_cost /= 60.0
    t4 = time.time()
    tc = (t2 - t1 + t4 - t3) / 60.0
    tb = (t3 - t2) / 60.0
    logger.info("get_avg_cost_mins calc time %f barrier time %f" % (tc, tb))
    return avg_cost


def get_max_cost_mins(value):
    #from mpi4py import MPI
    local_cost = np.array([value])
    global_cost = np.copy(local_cost) * 0
    #fleet._role_maker._node_type_comm.Allreduce(local_cost, global_cost, op=MPI.MAX)
    global_cost = fleet.util.all_reduce(local_cost, mode="max")
    logger.info("max train time %f mins" % (float(global_cost[0]) / 60.0))
    logger.info("max train time: %f", global_cost[0])


def get_min_cost_mins(value):
    #from mpi4py import MPI
    local_cost = np.array([value])
    global_cost = np.copy(local_cost) * 0
    #fleet._role_maker._node_type_comm.Allreduce(local_cost, global_cost, op=MPI.MIN)
    global_cost = fleet.util.all_reduce(local_cost, mode="min")
    logger.info("min train time %f mins" % (float(global_cost[0]) / 60.0))
