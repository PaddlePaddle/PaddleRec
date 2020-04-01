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
import os
import time
import numpy as np
import logging
import paddle.fluid as fluid

from .trainer import Trainer
from ..utils import envs

from ..reader import dataset

from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory
from paddle.fluid.incubate.fleet.base.role_maker import PaddleCloudRoleMaker

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def need_save(epoch_id, epoch_interval, is_last=False):
    if is_last:
        return True

    return epoch_id % epoch_interval == 0


class ClusterTrainer(Trainer):

    def __init__(self, config=None, yaml_file=None):
        Trainer.__init__(self, config, yaml_file)

        self.exe = fluid.Executor(fluid.CPUPlace())

        self.regist_context_processor('uninit', self.instance)
        self.regist_context_processor('init_pass', self.init)
        self.regist_context_processor('server_pass', self.server)
        self.regist_context_processor('train_pass', self.train)
        self.regist_context_processor('terminal_pass', self.terminal)

    def build_role_maker(self):
        role_maker = envs.get_global_env("train.role_maker")

        if role_maker == "PaddleCloudRoleMaker":
            role = PaddleCloudRoleMaker()
            return role
        else:
            raise ValueError("only support PaddleCloudRoleMaker now")

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

        return strategy

    def instance(self, context):
        model_package = __import__(envs.get_global_env("train.model.models"))
        train_model = getattr(model_package, 'Train')

        self.model = train_model()

        context['status'] = 'init_pass'

    def init(self, context):
        fleet.init(self.build_role_maker())

        self.model.input()
        self.model.net()
        self.model.loss()
        self.metrics = self.model.metrics()
        self.loss = self.model.avg_loss()

        optimizer = self.model.get_optimizer()
        strategy = self.build_strategy()
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self.loss)

        if fleet.is_server():
            context['status'] = 'server_pass'
        else:
            context['status'] = 'train_pass'

    def server(self, context):
        fleet.init_server()
        fleet.run_server()

        context['status'] = 'wait'

    def terminal(self, context):
        fleet.stop_worker()
        context['is_exit'] = True

    def train(self, context):
        print("Need to be implement")
        context['is_exit'] = True


class ClusterTrainerWithDataloader(ClusterTrainer):
    pass


class ClusterTrainerWithDataset(ClusterTrainer):
    def _get_dataset(self, inputs, threads, batch_size, pipe_command, train_files_path):
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var(inputs)
        dataset.set_pipe_command(pipe_command)
        dataset.set_batch_size(batch_size)
        dataset.set_thread(threads)
        file_list = [
            os.path.join(train_files_path, x)
            for x in os.listdir(train_files_path)
        ]

        dataset.set_filelist(file_list)
        return dataset

    def save(self, epoch_id):
        def save_inference_model():
            is_save_inference = envs.get_global_env("save.inference", False)
            if not is_save_inference:
                return

            save_interval = envs.get_global_env("save.inference.epoch_interval", 1)
            if not need_save(epoch_id, save_interval, False):
                return

            feed_varnames = envs.get_global_env("save.inference.feed_varnames", None)
            fetch_varnames = envs.get_global_env("save.inference.fetch_varnames", None)
            fetch_vars = [fluid.global_scope().vars[varname] for varname in fetch_varnames]
            dirname = envs.get_global_env("save.inference.dirname", None)

            assert dirname is not None
            dirname = os.path.join(dirname, str(epoch_id))
            fluid.io.save_inference_model(dirname, feed_varnames, fetch_vars, self.exe)

        def save_persistables():
            is_save_increment = envs.get_global_env("save.increment", False)
            if not is_save_increment:
                return

            save_interval = envs.get_global_env("save.increment.epoch_interval", 1)
            if not need_save(epoch_id, save_interval, False):
                return

            dirname = envs.get_global_env("save.inference.dirname", None)

            assert dirname is not None
            dirname = os.path.join(dirname, str(epoch_id))
            fluid.io.save_persistables(self.exe, dirname)

        is_save = envs.get_global_env("save", False)

        if not is_save:
            return

        save_persistables()
        save_inference_model()

    def train(self, context):
        inputs = self.model.input_vars()
        threads = envs.get_global_env("threads")
        batch_size = envs.get_global_env("batch_size")
        pipe_command = envs.get_global_env("pipe_command")
        train_data_path = envs.get_global_env("train_data_path")

        dataset = self._get_dataset(inputs, threads, batch_size, pipe_command, train_data_path)

        fleet.init_worker()
        self.exe.run(fleet.startup_program)

        epochs = envs.get_global_env("epochs")

        for i in range(epochs):
            self.exe.train_from_dataset(program=fluid.default_main_program(),
                                        dataset=dataset,
                                        fetch_list=[self.metrics],
                                        fetch_info=["epoch {} auc ".format(i)],
                                        print_period=100)
            self.save(i)

        context['status'] = 'infer_pass'

    def infer(self, context):
        context['status'] = 'terminal_pass'
