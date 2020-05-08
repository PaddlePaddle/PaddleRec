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

from fleetrec.core.trainers.transpiler_trainer import TranspileTrainer
from fleetrec.core.utils import envs
import numpy as np

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


class SingleTrainer(TranspileTrainer):
    def processor_register(self):
        self.regist_context_processor('uninit', self.instance)
        self.regist_context_processor('init_pass', self.init)

        if envs.get_platform() == "LINUX":
            self.regist_context_processor('train_pass', self.dataset_train)
        else:
            self.regist_context_processor('train_pass', self.dataloader_train)

        self.regist_context_processor('infer_pass', self.infer)
        self.regist_context_processor('terminal_pass', self.terminal)

    def init(self, context):
        self.model.train_net()
        optimizer = self.model.optimizer()
        optimizer.minimize((self.model.get_cost_op()))

        self.fetch_vars = []
        self.fetch_alias = []
        self.fetch_period = self.model.get_fetch_period()

        metrics = self.model.get_metrics()
        if metrics:
            self.fetch_vars = metrics.values()
            self.fetch_alias = metrics.keys()
        context['status'] = 'train_pass'

    def dataloader_train(self, context):
        self._exe.run(fluid.default_startup_program())
        reader = self._get_dataloader("TRAIN")
        epochs = envs.get_global_env("train.epochs")

        program = fluid.compiler.CompiledProgram(
            fluid.default_main_program()).with_data_parallel(
            loss_name=self.model.get_cost_op().name)

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
                    metrics_rets = self._exe.run(
                        program=program,
                        fetch_list=metrics_varnames)

                    metrics = [epoch, batch_id]
                    metrics.extend(metrics_rets)

                    if batch_id % 10 == 0 and batch_id != 0:
                        print(metrics_format.format(*metrics))
                    batch_id += 1
            except fluid.core.EOFException:
                reader.reset()
            self.save(epoch, "train", is_fleet=False)

        context['status'] = 'infer_pass'

    def dataset_train(self, context):
        # run startup program at once
        self._exe.run(fluid.default_startup_program())
        dataset = self._get_dataset("TRAIN")
        epochs = envs.get_global_env("train.epochs")

        for i in range(epochs):
            self._exe.train_from_dataset(program=fluid.default_main_program(),
                                         dataset=dataset,
                                         fetch_list=self.fetch_vars,
                                         fetch_info=self.fetch_alias,
                                         print_period=1,
                                         debug=True)
            self.save(i, "train", is_fleet=False)
        context['status'] = 'infer_pass'

    def infer(self, context):
        infer_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(infer_program, startup_program):
                self.model.infer_net()

        reader = self._get_dataloader("Evaluate")

        metrics_varnames = []
        metrics_format = []

        metrics_format.append("{}: {{}}".format("epoch"))
        metrics_format.append("{}: {{}}".format("batch"))

        for name, var in self.model.get_infer_results().items():
            metrics_varnames.append(var.name)
            metrics_format.append("{}: {{}}".format(name))

        metrics_format = ", ".join(metrics_format)
        self._exe.run(startup_program)

        for (epoch, model_dir) in self.increment_models:
            print("Begin to infer epoch {}, model_dir: {}".format(epoch, model_dir))
            program = infer_program.clone()
            fluid.io.load_persistables(self._exe, model_dir, program)
            reader.start()
            batch_id = 0
            try:
                while True:
                    metrics_rets = self._exe.run(
                        program=program,
                        fetch_list=metrics_varnames)

                    metrics = [epoch, batch_id]
                    metrics.extend(metrics_rets)

                    if batch_id % 2 == 0 and batch_id != 0:
                        print(metrics_format.format(*metrics))
                    batch_id += 1
            except fluid.core.EOFException:
                reader.reset()
 
        context['status'] = 'terminal_pass'

    def terminal(self, context):
        for model in self.increment_models:
            print("epoch :{}, dir: {}".format(model[0], model[1]))
        context['is_exit'] = True
