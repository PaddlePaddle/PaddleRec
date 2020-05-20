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

import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory
from paddle.fluid.incubate.fleet.base.role_maker import PaddleCloudRoleMaker

from paddlerec.core.utils import envs
from paddlerec.core.trainers.transpiler_trainer import TranspileTrainer


class ClusterTrainer(TranspileTrainer):
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
            self.regist_context_processor('startup_pass', self.startup)

            if envs.get_platform() == "LINUX" and envs.get_global_env(
                    "dataset_class", None, "train.reader") != "DataLoader":
                self.regist_context_processor('train_pass', self.dataset_train)
            else:
                self.regist_context_processor('train_pass',
                                              self.dataloader_train)

            self.regist_context_processor('infer_pass', self.infer)
            self.regist_context_processor('terminal_pass', self.terminal)

    def build_strategy(self):
        mode = envs.get_runtime_environ("train.trainer.strategy")
        assert mode in ["async", "geo", "sync", "half_async"]

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

        self.strategy = strategy
        return strategy

    def init(self, context):
        self.model.train_net()
        optimizer = self.model.optimizer()
        optimizer_name = envs.get_global_env("hyper_parameters.optimizer",
                                             None, "train.model")
        if optimizer_name not in ["", "sgd", "SGD", "Sgd"]:
            os.environ["FLAGS_communicator_is_sgd_optimizer"] = '0'

        strategy = self.build_strategy()
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self.model.get_avg_cost())

        if fleet.is_server():
            context['status'] = 'server_pass'
        else:
            self.fetch_vars = []
            self.fetch_alias = []
            self.fetch_period = self.model.get_fetch_period()

            metrics = self.model.get_metrics()
            if metrics:
                self.fetch_vars = metrics.values()
                self.fetch_alias = metrics.keys()
            context['status'] = 'startup_pass'

    def server(self, context):
        fleet.init_server()
        fleet.run_server()
        context['is_exit'] = True

    def startup(self, context):
        self._exe.run(fleet.startup_program)
        context['status'] = 'train_pass'

    def dataloader_train(self, context):
        fleet.init_worker()

        reader = self._get_dataloader()
        epochs = envs.get_global_env("train.epochs")

        program = fluid.compiler.CompiledProgram(
            fleet.main_program).with_data_parallel(
                loss_name=self.model.get_avg_cost().name,
                build_strategy=self.strategy.get_build_strategy(),
                exec_strategy=self.strategy.get_execute_strategy())

        metrics_varnames = []
        metrics_format = []

        metrics_format.append("{}: {{}}".format("epoch"))
        metrics_format.append("{}: {{}}".format("batch"))

        for name, var in self.model.get_metrics().items():
            metrics_varnames.append(var.name)
            metrics_format.append("{}: {{}}".format(name))

        metrics_format = ", ".join(metrics_format)

        for epoch in range(epochs):
            reader.start()
            batch_id = 0
            try:
                while True:
                    metrics_rets = self._exe.run(program=program,
                                                 fetch_list=metrics_varnames)

                    metrics = [epoch, batch_id]
                    metrics.extend(metrics_rets)

                    if batch_id % self.fetch_period == 0 and batch_id != 0:
                        print(metrics_format.format(*metrics))
                    batch_id += 1
            except fluid.core.EOFException:
                reader.reset()
            self.save(epoch, "train", is_fleet=True)

        fleet.stop_worker()
        context['status'] = 'infer_pass'

    def dataset_train(self, context):
        fleet.init_worker()

        dataset = self._get_dataset()
        ins = self._get_dataset_ins()

        epochs = envs.get_global_env("train.epochs")

        for i in range(epochs):
            begin_time = time.time()
            self._exe.train_from_dataset(
                program=fluid.default_main_program(),
                dataset=dataset,
                fetch_list=self.fetch_vars,
                fetch_info=self.fetch_alias,
                print_period=self.fetch_period)
            end_time = time.time()
            times = end_time - begin_time
            print("epoch {} using time {}, speed {:.2f} lines/s".format(
                i, times, ins / times))

            self.save(i, "train", is_fleet=True)
        fleet.stop_worker()
        context['status'] = 'infer_pass'

    def terminal(self, context):
        for model in self.increment_models:
            print("epoch :{}, dir: {}".format(model[0], model[1]))
        context['is_exit'] = True
