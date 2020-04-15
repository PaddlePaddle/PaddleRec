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

import paddle.fluid as fluid

from fleetrec.trainer.transpiler_trainer import TranspileTrainer
from fleetrec.utils import envs


class UserDefineTrainer(TranspileTrainer):
    def __init__(self, config=None):
        TranspileTrainer.__init__(self, config)
        print("this is a demo about how to use user define trainer in fleet-rec")

    def processor_register(self):
        self.regist_context_processor('uninit', self.instance)
        self.regist_context_processor('init_pass', self.init)
        self.regist_context_processor('train_pass', self.train)

    def init(self, context):
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

        context['is_exit'] = True
