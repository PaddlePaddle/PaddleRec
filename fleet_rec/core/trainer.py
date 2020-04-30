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
import sys

import abc
import time
import yaml

from paddle import fluid
from fleetrec.core.utils import envs


class Trainer(object):
    """R
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, config=None):
        self._status_processor = {}
        self._place = fluid.CPUPlace()
        self._exe = fluid.Executor(self._place)
        self._exector_context = {}
        self._context = {'status': 'uninit', 'is_exit': False}
        self._config_yaml = config

        with open(config, 'r') as rb:
            self._config = yaml.load(rb.read(), Loader=yaml.FullLoader)

    def regist_context_processor(self, status_name, processor):
        """
        regist a processor for specify status
        """
        self._status_processor[status_name] = processor

    def context_process(self, context):
        """
        select a processor to deal specify context
        Args:
            context : context with status
        Return:
            None : run a processor for this status
        """
        if context['status'] in self._status_processor:
            self._status_processor[context['status']](context)
        else:
            self.other_status_processor(context)

    def other_status_processor(self, context):
        """
        if no processor match context.status, use defalut processor
        Return:
            None, just sleep in base
        """
        print('unknow context_status:%s, do nothing' % context['status'])
        time.sleep(60)

    def reload_train_context(self):
        """
        context maybe update timely, reload for update
        """
        pass

    def run(self):
        """
        keep running by statu context.
        """
        while True:
            self.reload_train_context()
            self.context_process(self._context)
            if self._context['is_exit']:
                break


def user_define_engine(engine_yaml):
    with open(engine_yaml, 'r') as rb:
        _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
    assert _config is not None

    envs.set_runtime_environs(_config)

    train_location = envs.get_global_env("engine.file")
    train_dirname = os.path.dirname(train_location)
    base_name = os.path.splitext(os.path.basename(train_location))[0]
    sys.path.append(train_dirname)
    trainer_class = envs.lazy_instance_by_fliename(base_name, "UserDefineTraining")
    return trainer_class
