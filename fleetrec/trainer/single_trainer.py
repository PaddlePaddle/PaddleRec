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

from fleetrec.trainer.transpiler_trainer import TranspileTrainer
from fleetrec.utils import envs

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


class SingleTrainer(TranspileTrainer):
    def processor_register(self):
        self.regist_context_processor('uninit', self.instance)
        self.regist_context_processor('init_pass', self.init)
        self.regist_context_processor('train_pass', self.train)
        self.regist_context_processor('infer_pass', self.infer)
        self.regist_context_processor('terminal_pass', self.terminal)

    def init(self, context):
        self.model.input()
        self.model.net()
        self.model.metrics()
        self.model.avg_loss()
        optimizer = self.model.optimizer()
        optimizer.minimize(self.model._cost)

        self.fetch_vars = []
        self.fetch_alias = []
        self.fetch_period = self.model.get_fetch_period()

        metrics = self.model.get_metrics()
        if metrics:
            self.fetch_vars = metrics.values()
            self.fetch_alias = metrics.keys()
        context['status'] = 'train_pass'

    def train(self, context):
        # run startup program at once
        self._exe.run(fluid.default_startup_program())

        dataset = self._get_dataset()

        epochs = envs.get_global_env("train.epochs")

        for i in range(epochs):
            self._exe.train_from_dataset(program=fluid.default_main_program(),
                                         dataset=dataset,
                                         fetch_list=self.fetch_vars,
                                         fetch_info=self.fetch_alias,
                                         print_period=self.fetch_period)
            self.save(i, "train", is_fleet=False)
        context['status'] = 'infer_pass'

    def infer(self, context):
        context['status'] = 'terminal_pass'

    def terminal(self, context):
        for model in self.increment_models:
            print("epoch :{}, dir: {}".format(model[0], model[1]))
        context['is_exit'] = True
