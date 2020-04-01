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

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def need_save(epoch_id, epoch_interval, is_last=False):
    if is_last:
        return True

    return epoch_id % epoch_interval == 0


class SingleTrainer(Trainer):
    def __init__(self, config=None):
        Trainer.__init__(self, config)

        self.exe = fluid.Executor(fluid.CPUPlace())

        self.regist_context_processor('uninit', self.instance)
        self.regist_context_processor('init_pass', self.init)
        self.regist_context_processor('train_pass', self.train)
        self.regist_context_processor('infer_pass', self.infer)
        self.regist_context_processor('terminal_pass', self.terminal)

    def instance(self, context):
        model_package = __import__(envs.get_global_env("train.model.models"))
        train_model = getattr(model_package, 'Train')

        self.model = train_model()

        context['status'] = 'init_pass'

    def init(self, context):
        self.model.input()
        self.model.net()
        self.metrics = self.model.metrics()
        loss = self.model.avg_loss()

        optimizer = self.model.get_optimizer()
        optimizer.minimize(loss)

        # run startup program at once
        self.exe.run(fluid.default_startup_program())

        context['status'] = 'train_pass'

    def train(self, context):
        print("Need to be implement")
        context['is_exit'] = True

    def infer(self, context):
        print("Need to be implement")
        context['is_exit'] = True

    def terminal(self, context):
        context['is_exit'] = True


class SingleTrainerWithDataloader(SingleTrainer):
    pass


class SingleTrainerWithDataset(SingleTrainer):
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

        epochs = envs.get_global_env("epochs")

        for i in range(epochs):
            self.exe.train_from_dataset(program=fluid.default_main_program(),
                                        dataset=dataset,
                                        fetch_list=[self.metrics],
                                        fetch_info=["epoch {} auc ".format(i)],
                                        print_period=100)
        context['status'] = 'infer_pass'


def infer(self, context):
    context['status'] = 'terminal_pass'
