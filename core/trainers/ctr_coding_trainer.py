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
import os
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
from paddle.fluid.incubate.fleet.base.role_maker import MPISymetricRoleMaker

from paddlerec.core.utils import envs
from paddlerec.core.trainer import Trainer


class CtrPaddleTrainer(Trainer):
    """R
    """

    def __init__(self, config):
        """R
        """
        Trainer.__init__(self, config)

        self.global_config = config
        self._metrics = {}
        self.processor_register()

    def processor_register(self):
        role = MPISymetricRoleMaker()
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

    def _get_dataset(self):
        namespace = "train.reader"

        inputs = self.model.get_inputs()
        threads = envs.get_global_env("train.threads", None)
        batch_size = envs.get_global_env("batch_size", None, namespace)
        reader_class = envs.get_global_env("class", None, namespace)
        abs_dir = os.path.dirname(os.path.abspath(__file__))
        reader = os.path.join(abs_dir, '../utils', 'dataset_instance.py')
        pipe_cmd = "python {} {} {} {}".format(reader, reader_class, "TRAIN", self._config_yaml)
        train_data_path = envs.get_global_env("train_data_path", None, namespace)

        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var(inputs)
        dataset.set_pipe_command(pipe_cmd)
        dataset.set_batch_size(batch_size)
        dataset.set_thread(threads)
        file_list = [
            os.path.join(train_data_path, x)
            for x in os.listdir(train_data_path)
        ]

        dataset.set_filelist(file_list)
        return dataset

    def instance(self, context):
        models = envs.get_global_env("train.model.models")
        model_class = envs.lazy_instance_by_fliename(models, "Model")
        self.model = model_class(None)
        context['status'] = 'init_pass'

    def init(self, context):
        """R
        """
        self.model.train_net()
        optimizer = self.model.optimizer()

        optimizer = fleet.distributed_optimizer(optimizer, strategy={"use_cvm": False})
        optimizer.minimize(self.model.get_cost_op())

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
            context['status'] = 'train_pass'

    def server(self, context):
        fleet.run_server()
        fleet.stop_worker()
        context['is_exit'] = True

    def train(self, context):
        self._exe.run(fluid.default_startup_program())
        fleet.init_worker()

        dataset = self._get_dataset()

        shuf = np.array([fleet.worker_index()])
        gs = shuf * 0
        fleet._role_maker._node_type_comm.Allreduce(shuf, gs)

        print("trainer id: {}, trainers: {}, gs: {}".format(fleet.worker_index(), fleet.worker_num(), gs))

        epochs = envs.get_global_env("train.epochs")

        for i in range(epochs):
            self._exe.train_from_dataset(program=fluid.default_main_program(),
                                         dataset=dataset,
                                         fetch_list=self.fetch_vars,
                                         fetch_info=self.fetch_alias,
                                         print_period=self.fetch_period)

        context['status'] = 'terminal_pass'
        fleet.stop_worker()

    def terminal(self, context):
        print("terminal ended.")
        context['is_exit'] = True
