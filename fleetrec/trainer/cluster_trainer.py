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
Training use fluid with one node only.
"""

from __future__ import print_function
import logging

import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory
from paddle.fluid.incubate.fleet.base.role_maker import PaddleCloudRoleMaker

from ..utils import envs
from .transpiler_trainer import TranspileTrainer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


class ClusterTrainerWithDataloader(TranspileTrainer):
    pass


class ClusterTrainerWithDataset(TranspileTrainer):
    def processor_register(self):
        role = PaddleCloudRoleMaker()
        fleet.init(role)

        if fleet.is_server():
            self.regist_context_processor('uninit', self.instance)
            self.regist_context_processor('init_pass', self.init)
            self.regist_context_processor('server_pass', self.server)
        else:
            self.regist_context_processor('uninit', self.instance)
            self.regist_context_processor('init_pass', self.init)
            self.regist_context_processor('train_pass', self.train)
            self.regist_context_processor('terminal_pass', self.terminal)

    def build_strategy(self):
        mode = envs.get_global_env("train.strategy.mode")
        strategy = None

        if mode == "async":
            strategy = StrategyFactory.create_async_strategy()
        elif mode == "geo":
            push_num = envs.get_global_env("train.strategy.mode.push_num", 100)
            strategy = StrategyFactory.create_geo_strategy(push_num)
        elif mode == "sync":
            strategy = StrategyFactory.create_sync_strategy()
        elif mode == "half_async":
            strategy = StrategyFactory.create_half_async_strategy()

        assert strategy is not None

        return strategy

    def init(self, context):

        print("init pass")

        self.model.input()
        self.model.net()
        self.metrics = self.model.metrics()
        self.metric_extras = self.model.metric_extras()

        loss = self.model.avg_loss()
        optimizer = self.model.optimizer()

        strategy = self.build_strategy()
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(loss)

        if fleet.is_server():
            context['status'] = 'server_pass'
        else:
            context['status'] = 'train_pass'

    def server(self, context):
        fleet.init_server()
        fleet.run_server()
        context['is_exit'] = True

    def terminal(self, context):
        fleet.stop_worker()
        context['is_exit'] = True

    def train(self, context):
        self.exe.run(fleet.startup_program)
        fleet.init_worker()

        dataset = self._get_dataset()
        epochs = envs.get_global_env("train.epochs")

        for i in range(epochs):
            self.exe.train_from_dataset(program=fluid.default_main_program(),
                                        dataset=dataset,
                                        fetch_list=self.metric_extras[0],
                                        fetch_info=self.metric_extras[1],
                                        print_period=self.metric_extras[2])
            self.save(i, "train", is_fleet=True)
        context['status'] = 'infer_pass'
        fleet.stop_worker()

    def infer(self, context):
        context['status'] = 'terminal_pass'

    def terminal(self, context):
        for model in self.increment_models:
            print("epoch :{}, dir: {}".format(model[0], model[1]))
        context['is_exit'] = True
