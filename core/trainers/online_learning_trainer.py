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

import datetime
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
            if envs.get_platform() == "LINUX" and envs.get_global_env("dataset_class", None, "train.reader") != "DataLoader":
                self.regist_context_processor('train_pass', self.dataset_train)
            else:
                self.regist_context_processor(
                    'train_pass', self.dataloader_train)
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
        print("online learning can only support LINUX only")
        context['status'] = 'terminal_pass'

    def _get_dataset(self, state="TRAIN", hour=None):
        if state == "TRAIN":
            inputs = self.model.get_inputs()
            namespace = "train.reader"
            train_data_path = envs.get_global_env(
                "train_data_path", None, namespace)
        else:
            inputs = self.model.get_infer_inputs()
            namespace = "evaluate.reader"
            train_data_path = envs.get_global_env(
                "test_data_path", None, namespace)

        threads = int(envs.get_runtime_environ("train.trainer.threads"))
        batch_size = envs.get_global_env("batch_size", None, namespace)
        reader_class = envs.get_global_env("class", None, namespace)
        abs_dir = os.path.dirname(os.path.abspath(__file__))
        reader = os.path.join(abs_dir, '../utils', 'dataset_instance.py')
        pipe_cmd = "python {} {} {} {}".format(
            reader, reader_class, state, self._config_yaml)

        if train_data_path.startswith("paddlerec::"):
            package_base = envs.get_runtime_environ("PACKAGE_BASE")
            assert package_base is not None
            train_data_path = os.path.join(
                package_base, train_data_path.split("::")[1])

        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var(inputs)
        dataset.set_pipe_command(pipe_cmd)
        dataset.set_batch_size(batch_size)
        dataset.set_thread(threads)

        if hour is not None:
            train_data_path = os.path.join(train_data_path, hour)

        file_list = [
            os.path.join(train_data_path, x)
            for x in os.listdir(train_data_path)
        ]

        self.files = file_list
        dataset.set_filelist(self.files)
        return dataset

    def dataset_train(self, context):
        fleet.init_worker()

        days = envs.get_global_env("train.days")
        begin_day = datetime.datetime.strptime("begin_day_d", '%Y%m%d')

        for day in range(days):
            for hour in range(24):
                day = begin_day + datetime.timedelta(days=day, hours=hour)
                day_s = day.strftime('%Y%m%d/%H')
                i = day.strftime('%Y%m%d_%H')

                dataset = self._get_dataset(hour=day_s)
                ins = self._get_dataset_ins()

                begin_time = time.time()
                self._exe.train_from_dataset(program=fluid.default_main_program(),
                                             dataset=dataset,
                                             fetch_list=self.fetch_vars,
                                             fetch_info=self.fetch_alias,
                                             print_period=self.fetch_period)
                end_time = time.time()
                times = end_time-begin_time
                print("epoch {} using time {}, speed {:.2f} lines/s".format(i, times, ins/times))
                self.save(i, "train", is_fleet=True)

        fleet.stop_worker()
        context['status'] = 'infer_pass'

    def terminal(self, context):
        for model in self.increment_models:
            print("epoch :{}, dir: {}".format(model[0], model[1]))
        context['is_exit'] = True
