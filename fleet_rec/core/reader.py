#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function

import abc
import os

import paddle.fluid.incubate.data_generator as dg
import yaml

from fleet_rec.core.utils import envs


class Reader(dg.MultiSlotDataGenerator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        dg.MultiSlotDataGenerator.__init__(self)

        if os.path.isfile(config):
            with open(config, 'r') as rb:
                _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
        else:
            raise ValueError("reader config only support yaml")

        envs.set_global_envs(_config)

    @abc.abstractmethod
    def init(self):
        pass

    @abc.abstractmethod
    def generate_sample(self, line):
        pass
