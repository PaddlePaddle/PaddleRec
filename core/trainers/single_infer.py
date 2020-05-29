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

import time
import logging
import os
import paddle.fluid as fluid

from paddlerec.core.trainers.transpiler_trainer import TranspileTrainer
from paddlerec.core.utils import envs
from paddlerec.core.reader import SlotReader
from paddlerec.core.utils import dataloader_instance

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


class SingleInfer(TranspileTrainer):
    def __init__(self, config=None):
        super(TranspileTrainer, self).__init__(config)
        self._env = self._config
        device = envs.get_global_env("device")
        if device == 'gpu':
            self._place = fluid.CUDAPlace(0)
        elif device == 'cpu':
            self._place = fluid.CPUPlace()
        self._exe = fluid.Executor(self._place)
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
        self.regist_context_processor('uninit', self.instance)
        self.regist_context_processor('init_pass', self.init)
        self.regist_context_processor('startup_pass', self.startup)
        self.regist_context_processor('train_pass', self.executor_train)
        self.regist_context_processor('terminal_pass', self.terminal)

    def instance(self, context):
        context['status'] = 'init_pass'

    def _get_dataset(self, dataset_name):
        name = "dataset." + dataset_name + "."
        thread_num = envs.get_global_env(name + "thread_num")
        batch_size = envs.get_global_env(name + "batch_size")
        reader_class = envs.get_global_env(name + "data_converter")
        abs_dir = os.path.dirname(os.path.abspath(__file__))
        reader = os.path.join(abs_dir, '../utils', 'dataset_instance.py')
        sparse_slots = envs.get_global_env(name + "sparse_slots", "").strip()
        dense_slots = envs.get_global_env(name + "dense_slots", "").strip()
        if sparse_slots == "" and dense_slots == "":
            pipe_cmd = "python {} {} {} {}".format(reader, reader_class,
                                                   "TRAIN", self._config_yaml)
        else:
            if sparse_slots is None:
                sparse_slots = "#"
            if dense_slots is None:
                dense_slots = "#"
            padding = envs.get_global_env(name + "padding", 0)
            pipe_cmd = "python {} {} {} {} {} {} {} {}".format(
                reader, "slot", "slot", self._config_yaml, "fake", \
                sparse_slots.replace(" ", "#"), dense_slots.replace(" ", "#"), str(padding))

        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_batch_size(envs.get_global_env(name + "batch_size"))
        dataset.set_pipe_command(pipe_cmd)
        train_data_path = envs.get_global_env(name + "data_path")
        file_list = [
            os.path.join(train_data_path, x)
            for x in os.listdir(train_data_path)
        ]
        dataset.set_filelist(file_list)
        for model_dict in self._env["phase"]:
            if model_dict["dataset_name"] == dataset_name:
                model = self._model[model_dict["name"]][3]
                inputs = model._infer_data_var
                dataset.set_use_var(inputs)
                break
        return dataset

    def _get_dataloader(self, dataset_name, dataloader):
        name = "dataset." + dataset_name + "."
        thread_num = envs.get_global_env(name + "thread_num")
        batch_size = envs.get_global_env(name + "batch_size")
        reader_class = envs.get_global_env(name + "data_converter")
        abs_dir = os.path.dirname(os.path.abspath(__file__))
        sparse_slots = envs.get_global_env(name + "sparse_slots", "").strip()
        dense_slots = envs.get_global_env(name + "dense_slots", "").strip()
        if sparse_slots == "" and dense_slots == "":
            reader = dataloader_instance.dataloader_by_name(
                reader_class, dataset_name, self._config_yaml)
            reader_class = envs.lazy_instance_by_fliename(reader_class,
                                                          "TrainReader")
            reader_ins = reader_class(self._config_yaml)
        else:
            reader = dataloader_instance.slotdataloader_by_name(
                "", dataset_name, self._config_yaml)
            reader_ins = SlotReader(self._config_yaml)
        if hasattr(reader_ins, 'generate_batch_from_trainfiles'):
            dataloader.set_sample_list_generator(reader)
        else:
            dataloader.set_sample_generator(reader, batch_size)
        return dataloader

    def _create_dataset(self, dataset_name):
        name = "dataset." + dataset_name + "."
        sparse_slots = envs.get_global_env(name + "sparse_slots")
        dense_slots = envs.get_global_env(name + "dense_slots")
        thread_num = envs.get_global_env(name + "thread_num")
        batch_size = envs.get_global_env(name + "batch_size")
        type_name = envs.get_global_env(name + "type")
        if envs.get_platform() != "LINUX":
            print("platform ", envs.get_platform(),
                  " change reader to DataLoader")
            type_name = "DataLoader"
        padding = 0

        if type_name == "DataLoader":
            return None
        else:
            return self._get_dataset(dataset_name)

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
                        model._infer_data_var = model.input_data(
                            dataset_name=model_dict["dataset_name"])
                        if envs.get_global_env("dataset." + dataset_name +
                                               ".type") == "DataLoader":
                            model._init_dataloader(is_infer=True)
                            self._get_dataloader(dataset_name,
                                                 model._data_loader)
                        model.net(model._infer_data_var, True)
            self._model[model_dict["name"]][0] = train_program
            self._model[model_dict["name"]][1] = startup_program
            self._model[model_dict["name"]][2] = scope
            self._model[model_dict["name"]][3] = model
            self._model[model_dict["name"]][4] = train_program.clone()

        for dataset in self._env["dataset"]:
            if dataset["type"] != "DataLoader":
                self._dataset[dataset["name"]] = self._create_dataset(dataset[
                    "name"])

        context['status'] = 'startup_pass'

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
                                ".fetch_period", 20))
        metrics = model_class.get_infer_results()
        if metrics:
            fetch_vars = metrics.values()
            fetch_alias = metrics.keys()
        scope = self._model[model_name][2]
        program = self._model[model_name][0]
        reader = self._dataset[reader_name]
        with fluid.scope_guard(scope):
            self._exe.infer_from_dataset(
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
        fetch_vars = []
        fetch_alias = []
        metrics = model_class.get_infer_results()
        if metrics:
            fetch_vars = metrics.values()
            fetch_alias = metrics.keys()
        metrics_varnames = []
        metrics_format = []
        fetch_period = int(
            envs.get_global_env("runner." + self._runner_name +
                                ".fetch_period", 20))
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
        context['is_exit'] = True

    def load(self, is_fleet=False):
        name = "runner." + self._runner_name + "."
        dirname = envs.get_global_env("epoch.init_model_path", None)
        if dirname is None or dirname == "":
            return
        print("single_infer going to load ", dirname)
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
                name + "save_inference_feed_varnames", None)
            fetch_varnames = envs.get_global_env(
                name + "save_inference_fetch_varnames", None)
            if feed_varnames is None or fetch_varnames is None or feed_varnames == "":
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
