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

"""
Training use fluid with DistributeTranspiler
"""
from __future__ import print_function

import time
import warnings

import paddle.fluid as fluid
from paddlerec.core.utils import envs
from paddlerec.core.trainer import Trainer, EngineMode, FleetMode, Device

from .framework import dataset
from .framework import executor
from .framework import instance
from .framework import network
from .framework import startup


class GeneralTrainer(Trainer):
    def __init__(self, config=None):
        Trainer.__init__(self, config)
        self.processor_register()

    def processor_register(self):
        # single
        self.regist_context_processor('uninit', self.instance)
        self.regist_context_processor('network_pass', self.network)
        self.regist_context_processor('startup_pass', self.startup)
        self.regist_context_processor('train_pass', self.executor)
        self.regist_context_processor('terminal_pass', self.terminal)

    def instance(self, context):
        instance_class = None
        if self.engine == EngineMode.SINGLE:
            instance_class = instance.SingleInstance(context)
        elif self.fleet_mode == FleetMode.PS:
            instance_class = instance.PSInstance(context)
        elif self.fleet_mode == FleetMode.COLLECTIVE:
            instance_class = instance.CollectiveInstance(context)
        else:
            raise ValueError("Instance Init Error")
        instance_class.instance(context)

    def network(self, context):
        network_class = None
        if self.engine == EngineMode.SINGLE:
            network_class = network.SingleNetWork(context)
        elif self.fleet_mode == FleetMode.PS:
            network_class = network.PSNetwork(context)
        elif self.fleet_mode == FleetMode.COLLECTIVE:
            network_class = network.CollectiveNetWork(context)
        else:
            raise ValueError("NetWork Init Error")
        network_class.build_network(context)

    def startup(self, context):
        startup_class = None
        if self.engine == EngineMode.SINGLE:
            startup_class = startup.SingleStartup(context)
        elif self.fleet_mode == FleetMode.PS:
            startup_class = startup.PSStartUp(context)
        elif self.fleet_mode == FleetMode.COLLECTIVE:
            startup_class = startup.CollectiveStartUp(context)
        else:
            raise ValueError("Startup Init Error")
        startup_class.startup(context)

    def executor(self, context):
        executor_class = None
        if self.engine == EngineMode.SINGLE:
            executor_class = executor.SingleTrainExecutor(context)
        elif self.fleet_mode == FleetMode.PS:
            executor_class = executor.PSTrainExecutor(context)
        elif self.fleet_mode == FleetMode.COLLECTIVE:
            executor_class = executor.CollectiveTrainExecutor(context)
        else:
            raise ValueError("Executor Init Error")
        executor_class.exuctor(context)

    def terminal(self, context):
        context['is_exit'] = True
