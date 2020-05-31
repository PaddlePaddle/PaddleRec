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
    def __init__(self, config=None):
        super(TranspileTrainer, self).__init__(config)
        self._env = self._config
        self.processor_register()
        self._model = {}
        self._dataset = {}
        envs.set_global_envs(self._config)
        envs.update_workspace()
        self._runner_name = envs.get_global_env("mode")
        device = envs.get_global_env("runner." + self._runner_name + ".device")
        if device == 'gpu':
            self._place = fluid.CUDAPlace(0)
        elif device == 'cpu':
            self._place = fluid.CPUPlace()
        self._exe = fluid.Executor(self._place)

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
            self.regist_context_processor('train_pass', self.executor_train)
            self.regist_context_processor('infer_pass', self.infer)
            self.regist_context_processor('terminal_pass', self.terminal)

    def build_strategy(self):
        mode = envs.get_global_env(
            "runner." + self._runner_name + ".strategy")
        assert mode.upper() in ["ASYNC", "GEO", "SYNC", "HALF_ASYNC"]

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
        for model_dict in self._env["phase"]:
            self._model[model_dict["name"]] = [None] * 5
            train_program = fluid.Program()
            startup_program = fluid.Program()
            scope = fluid.Scope()
            dataset_name = model_dict["dataset_name"]
            opt_name = envs.get_global_env("hyper_parameters.optimizer.class")
            opt_lr = envs.get_global_env(
                "hyper_parameters.optimizer.learning_rate")
            opt_strategy = envs.get_global_env(
                "hyper_parameters.optimizer.strategy")
            with fluid.program_guard(train_program, startup_program):
                with fluid.unique_name.guard():
                    with fluid.scope_guard(scope):
                        model_path = model_dict["model"].replace(
                            "{workspace}",
                            envs.path_adapter(self._env["workspace"]))
                        model = envs.lazy_instance_by_fliename(
                            model_path, "Model")(self._env)
                        model._data_var = model.input_data(
                            dataset_name=model_dict["dataset_name"])
                        if envs.get_global_env("dataset." + dataset_name +
                                               ".type") == "DataLoader":
                            model._init_dataloader(is_infer=False)
                            self._get_dataloader(dataset_name,
                                                 model._data_loader)
                        model.net(model._data_var, False)
                        optimizer = model._build_optimizer(opt_name, opt_lr,
                                                           opt_strategy)
                        optimizer_name = envs.get_global_env("hyper_parameters.optimizer",
                                                             None, "optimizer.class")
                        if optimizer_name.upper() not in ["", "SGD"]:
                            os.environ["FLAGS_communicator_is_sgd_optimizer"] = '0'
                        strategy = self.build_strategy()
                        optimizer = fleet.distributed_optimizer(
                            optimizer, strategy)
                        optimizer.minimize(model._cost)
            self._model[model_dict["name"]][0] = train_program
            self._model[model_dict["name"]][1] = startup_program
            self._model[model_dict["name"]][2] = scope
            self._model[model_dict["name"]][3] = model
            self._model[model_dict["name"]][4] = train_program.clone()

        for dataset in self._env["dataset"]:
            if dataset["type"] != "DataLoader":
                self._dataset[dataset["name"]] = self._create_dataset(dataset[
                    "name"])

        if fleet.is_server():
            context['status'] = 'server_pass'
        else:
            context['status'] = 'startup_pass'

    def server(self, context):
        fleet.init_server()
        fleet.run_server()
        context['is_exit'] = True

    def startup(self, context):
        for model_dict in self._env["phase"]:
            with fluid.scope_guard(self._model[model_dict["name"]][2]):
                self._exe.run(self._model[model_dict["name"]][1])
        context['status'] = 'train_pass'

    def executor_train(self, context):
        epochs = int(
            envs.get_global_env("runner." + self._runner_name + ".epochs"))
        for j in range(epochs):
            for model_dict in self._env["phase"]:
                if j == 0:
                    with fluid.scope_guard(self._model[model_dict["name"]][2]):
                        train_prog = self._model[model_dict["name"]][0]
                        startup_prog = self._model[model_dict["name"]][1]
                        with fluid.program_guard(train_prog, startup_prog):
                            self.load()
                reader_name = model_dict["dataset_name"]
                name = "dataset." + reader_name + "."
                begin_time = time.time()
                if envs.get_global_env(name + "type") == "DataLoader":
                    self._executor_dataloader_train(model_dict)
                else:
                    self._executor_dataset_train(model_dict)
                with fluid.scope_guard(self._model[model_dict["name"]][2]):
                    train_prog = self._model[model_dict["name"]][4]
                    startup_prog = self._model[model_dict["name"]][1]
                    with fluid.program_guard(train_prog, startup_prog):
                        self.save(j)
                end_time = time.time()
                seconds = end_time - begin_time
            print("epoch {} done, time elasped: {}".format(j, seconds))
        context['status'] = "terminal_pass"

    def _executor_dataset_train(self, model_dict):
        reader_name = model_dict["dataset_name"]
        model_name = model_dict["name"]
        model_class = self._model[model_name][3]
        fetch_vars = []
        fetch_alias = []
        fetch_period = int(
            envs.get_global_env("runner." + self._runner_name +
                                ".print_interval", 20))
        metrics = model_class.get_metrics()
        if metrics:
            fetch_vars = metrics.values()
            fetch_alias = metrics.keys()
        scope = self._model[model_name][2]
        program = self._model[model_name][0]
        reader = self._dataset[reader_name]
        with fluid.scope_guard(scope):
            self._exe.train_from_dataset(
                program=program,
                dataset=reader,
                fetch_list=fetch_vars,
                fetch_info=fetch_alias,
                print_period=fetch_period)

    def _executor_dataloader_train(self, model_dict):
        reader_name = model_dict["dataset_name"]
        model_name = model_dict["name"]
        model_class = self._model[model_name][3]
        program = self._model[model_name][0].clone()
        program = fluid.compiler.CompiledProgram(program).with_data_parallel(
            loss_name=model_class.get_avg_cost().name)
        fetch_vars = []
        fetch_alias = []
        fetch_period = int(
            envs.get_global_env("runner." + self._runner_name +
                                ".print_interval", 20))
        metrics = model_class.get_metrics()
        if metrics:
            fetch_vars = metrics.values()
            fetch_alias = metrics.keys()
        metrics_varnames = []
        metrics_format = []
        metrics_format.append("{}: {{}}".format("batch"))
        for name, var in metrics.items():
            metrics_varnames.append(var.name)
            metrics_format.append("{}: {{}}".format(name))
        metrics_format = ", ".join(metrics_format)

        reader = self._model[model_name][3]._data_loader
        reader.start()
        batch_id = 0
        scope = self._model[model_name][2]
        with fluid.scope_guard(scope):
            try:
                while True:
                    metrics_rets = self._exe.run(program=program,
                                                 fetch_list=metrics_varnames)
                    metrics = [batch_id]
                    metrics.extend(metrics_rets)

                    if batch_id % fetch_period == 0 and batch_id != 0:
                        print(metrics_format.format(*metrics))
                    batch_id += 1
            except fluid.core.EOFException:
                reader.reset()

    def terminal(self, context):
        for model in self.increment_models:
            print("epoch :{}, dir: {}".format(model[0], model[1]))
        context['is_exit'] = True

    def load(self, is_fleet=False):
        dirname = envs.get_global_env(
            "runner." + self._runner_name + ".init_model_path", None)
        if dirname is None or dirname == "":
            return
        print("going to load ", dirname)
        if is_fleet:
            fleet.load_persistables(self._exe, dirname)
        else:
            fluid.io.load_persistables(self._exe, dirname)

    def save(self, epoch_id, is_fleet=False):
        def need_save(epoch_id, epoch_interval, is_last=False):
            if is_last:
                return True
            if epoch_id == -1:
                return False

            return epoch_id % epoch_interval == 0

        def save_inference_model():
            name = "runner." + self._runner_name + "."
            save_interval = int(
                envs.get_global_env(name + "save_inference_interval", -1))
            if not need_save(epoch_id, save_interval, False):
                return
            feed_varnames = envs.get_global_env(
                name + "save_inference_feed_varnames", [])
            fetch_varnames = envs.get_global_env(
                name + "save_inference_fetch_varnames", [])
            if feed_varnames is None or fetch_varnames is None or feed_varnames == "" or fetch_varnames == "" or \
               len(feed_varnames) == 0 or len(fetch_varnames) == 0:
                return
            fetch_vars = [
                fluid.default_main_program().global_block().vars[varname]
                for varname in fetch_varnames
            ]
            dirname = envs.get_global_env(name + "save_inference_path", None)

            assert dirname is not None
            dirname = os.path.join(dirname, str(epoch_id))

            if is_fleet:
                fleet.save_inference_model(self._exe, dirname, feed_varnames,
                                           fetch_vars)
            else:
                fluid.io.save_inference_model(dirname, feed_varnames,
                                              fetch_vars, self._exe)

        def save_persistables():
            name = "runner." + self._runner_name + "."
            save_interval = int(
                envs.get_global_env(name + "save_checkpoint_interval", -1))
            if not need_save(epoch_id, save_interval, False):
                return
            dirname = envs.get_global_env(name + "save_checkpoint_path", None)
            if dirname is None or dirname == "":
                return
            dirname = os.path.join(dirname, str(epoch_id))
            if is_fleet:
                fleet.save_persistables(self._exe, dirname)
            else:
                fluid.io.save_persistables(self._exe, dirname)

        save_persistables()
        save_inference_model()
