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

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


class SingleTrainer(TranspileTrainer):
    def __init__(self, config=None):
        super(TranspileTrainer, self).__init__(config)
        self._env = self._config#envs.get_global_envs()
        #device = envs.get_global_env("train.device", "cpu")
        device = self._env["device"]
        if device == 'gpu':
            self._place = fluid.CUDAPlace(0)
        elif device == 'cpu':
            self._place = fluid.CPUPlace()
        self._exe = fluid.Executor(self._place)
        self.processor_register()
        self._model = {}
        self._dataset = {}
        #self.inference_models = []
        #self.increment_models = []

    def processor_register(self):
        self.regist_context_processor('uninit', self.instance)
        self.regist_context_processor('init_pass', self.init)
        self.regist_context_processor('startup_pass', self.startup)

        #if envs.get_platform() == "LINUX" and envs.get_global_env(
        #        "dataset_class", None, "train.reader") != "DataLoader":

        self.regist_context_processor('train_pass', self.executor_train)
#        if envs.get_platform() == "LINUX" and envs.get_global_env(
#                 ""
#            self.regist_context_processor('train_pass', self.dataset_train)
#        else:
#            self.regist_context_processor('train_pass', self.dataloader_train)

        #self.regist_context_processor('infer_pass', self.infer)
        self.regist_context_processor('terminal_pass', self.terminal)

    def instance(self, context):
        context['status'] = 'init_pass'

    def dataloader_train(self, context):
        pass

    def dataset_train(self, context):
        pass

    #def _get_optmizer(self, cost):
    #    if self._env["hyper_parameters"]["optimizer"]["class"] == "Adam":
            
    def _create_dataset(self, dataset_name):
        config_dict = None
        for i in self._env["dataset"]:
            if i["name"] == dataset_name:
                config_dict = i
                break
        #reader_ins = SlotReader(self._config_yaml)
        sparse_slots = config_dict["sparse_slots"]
        dense_slots = config_dict["dense_slots"]
        padding = 0
        reader = envs.path_adapter("paddlerec.core.utils") + "/dataset_instance.py"
        #reader = "{workspace}/paddlerec/core/utils/dataset_instance.py".replace("{workspace}", envs.path_adapter(self._env["workspace"]))
        pipe_cmd = "python {} {} {} {} {} {} {} {}".format(
            reader, "slot", "slot", self._config_yaml, "fake", \
            sparse_slots.replace(" ", "#"), dense_slots.replace(" ", "#"), str(padding))

        if config_dict["type"] == "QueueDataset":
            dataset = fluid.DatasetFactory().create_dataset(config_dict["type"])
            dataset.set_batch_size(config_dict["batch_size"])
            #dataset.set_thread(config_dict["thread_num"])
            #dataset.set_hdfs_config(config_dict["data_fs_name"], config_dict["data_fs_ugi"])
            dataset.set_pipe_command(pipe_cmd)
            train_data_path = config_dict["data_path"].replace("{workspace}", envs.path_adapter(self._env["workspace"]))
            file_list = [
                os.path.join(train_data_path, x)
                for x in os.listdir(train_data_path)
            ]
            dataset.set_filelist(file_list)
            for model_dict in self._env["executor"]:
                if model_dict["dataset_name"] == dataset_name:
                    model = self._model[model_dict["name"]][3]
                    inputs = model.get_inputs()
                    dataset.set_use_var(inputs)
                    break
        else:
            pass

        return dataset

    def init(self, context):
        #self.model.train_net()
        for model_dict in self._env["executor"]:
            self._model[model_dict["name"]] = [None] * 4
#            self._model[model_dict["name"]][0] = fluid.Program() #train_program
#            self._model[model_dict["name"]][1] = fluid.Program() #startup_program
#            self._model[model_dict["name"]][2] = fluid.Scope()   #scope
            train_program = fluid.Program()
            startup_program = fluid.Program()
            scope = fluid.Scope()
            opt_name = self._env["hyper_parameters"]["optimizer"]["class"]
            opt_lr = self._env["hyper_parameters"]["optimizer"]["learning_rate"]
            opt_strategy = self._env["hyper_parameters"]["optimizer"]["strategy"]
            with fluid.program_guard(train_program, startup_program):
                with fluid.unique_name.guard():
                    model_path = model_dict["model"].replace("{workspace}", envs.path_adapter(self._env["workspace"]))
                    model = envs.lazy_instance_by_fliename(model_path, "Model")(self._env)
                    model._data_var = model.input_data(dataset_name=model_dict["dataset_name"])
                    model.net(None)####
                    optimizer = model._build_optimizer(opt_name, opt_lr, opt_strategy)
                    optimizer.minimize(model._cost)
            self._model[model_dict["name"]][0] = train_program
            self._model[model_dict["name"]][1] = startup_program
            self._model[model_dict["name"]][2] = scope
            self._model[model_dict["name"]][3] = model

        for dataset in self._env["dataset"]:
            self._dataset[dataset["name"]] = self._create_dataset(dataset["name"])

#        self.fetch_vars = []
#        self.fetch_alias = []
#        self.fetch_period = self.model.get_fetch_period()

#        metrics = self.model.get_metrics()
#        if metrics:
#            self.fetch_vars = metrics.values()
#            self.fetch_alias = metrics.keys()
        #evaluate_only = envs.get_global_env(
        #    'evaluate_only', False, namespace='evaluate')
        #if evaluate_only:
        #    context['status'] = 'infer_pass'
        #else:
        context['status'] = 'startup_pass'

    def startup(self, context):
        for model_dict in self._env["executor"]:
            with fluid.scope_guard(self._model[model_dict["name"]][2]):            
                self._exe.run(self._model[model_dict["name"]][1])
        context['status'] = 'train_pass'

    def executor_train(self, context):
        epochs = int(self._env["epochs"])
        for j in range(epochs):
            for model_dict in self._env["executor"]:
                reader_name = model_dict["dataset_name"]
                #print(self._dataset)
                #print(reader_name)
                dataset = None
                for i in self._env["dataset"]:
                    if i["name"] == reader_name:
                        dataset = i
                        break
                if dataset["type"] == "DataLoader":
                    self._executor_dataloader_train(model_dict)
                else:
                    self._executor_dataset_train(model_dict)
            print("epoch %s done" % j)
#                  self._model[model_dict["name"]][1] = fluid.compiler.CompiledProgram(
#                    self._model[model_dict["name"]][1]).with_data_parallel(loss_name=self._model.get_avg_cost().name)
#            fetch_vars = []
#            fetch_alias = []
#            fetch_period = self._model.get_fetch_period()
#            metrics = self._model.get_metrics()
#            if metrics:
#                fetch_vars = metrics.values()
#                fetch_alias = metrics.keys()
#            metrics_varnames = []
        context['status'] = "terminal_pass"

    def _executor_dataset_train(self, model_dict):
#        dataset = self._get_dataset("TRAIN")
#        ins = self._get_dataset_ins()

#        epochs = envs.get_global_env("train.epochs")
#        for i in range(epochs):
        reader_name = model_dict["dataset_name"]
        model_name = model_dict["name"]
        model_class = self._model[model_name][3]
        fetch_vars = []
        fetch_alias = []
        fetch_period = 1#model_class.get_fetch_period()
        metrics = model_class.get_metrics()
        if metrics:
            fetch_vars = metrics.values()
            fetch_alias = metrics.keys()
        scope = self._model[model_name][2]
        program = self._model[model_name][1]
        reader = self._dataset[reader_name]
        with fluid.scope_guard(scope):
            begin_time = time.time()
            self._exe.train_from_dataset(
                program=program,
                dataset=reader,
                fetch_list=fetch_vars,
                fetch_info=fetch_alias,
                print_period=fetch_period)
            end_time = time.time()
            times = end_time - begin_time
            #print("epoch {} using time {}".format(i, times))
            #print("epoch {} using time {}, speed {:.2f} lines/s".format(
            #    i, times, ins / times)) 


    def _executor_dataloader_train(self, model_dict):
        reader_name = model_dict["dataset_name"]
        model_name = model_dict["name"]
        model_class = self._model[model][3]
        self._model[model_name][1] = fluid.compiler.CompiledProgram(
            self._model[model_name][1]).with_data_parallel(loss_name=model_class.get_avg_cost().name)
        fetch_vars = []
        fetch_alias = []
        fetch_period = self._model.get_fetch_period()
        metrics = self._model.get_metrics()
        if metrics:
            fetch_vars = metrics.values()
            fetch_alias = metrics.keys()
        metrics_varnames = []
        metrics_format = []
        metrics_format.append("{}: {{}}".format("epoch"))
        metrics_format.append("{}: {{}}".format("batch"))
        for name, var in model_class.items():
            metrics_varnames.append(var.name)
            metrics_format.append("{}: {{}}".format(name))
        metrics_format = ", ".join(metrics_format)

        reader = self._dataset["reader_name"]
        reader.start()
        batch_id = 0
        scope = self._model[model_name][3]
        prorgram = self._model[model_name][1]
        with fluid.scope_guard(self._model[model_name][3]):
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


    def dataloader_train(self, context):
        
        exit(-1)

        reader = self._get_dataloader(self._env["TRAIN"])
        epochs = self._env["epochs"]

        program = fluid.compiler.CompiledProgram(fluid.default_main_program(
        )).with_data_parallel(loss_name=self.model.get_avg_cost().name)

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
            self.save(epoch, "train", is_fleet=False)

        context['status'] = 'infer_pass'

    def dataset_train(self, context):
        dataset = self._get_dataset("TRAIN")
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

            self.save(i, "train", is_fleet=False)
        context['status'] = 'infer_pass'

    def terminal(self, context):
        #for model in self.increment_models:
        #    print("epoch :{}, dir: {}".format(model[0], model[1]))
        context['is_exit'] = True
