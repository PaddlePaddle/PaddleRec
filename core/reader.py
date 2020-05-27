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

from paddlerec.core.utils import envs


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
        envs.update_workspace()

    @abc.abstractmethod
    def init(self):
        """init"""
        pass

    @abc.abstractmethod
    def generate_sample(self, line):
        pass


class SlotReader(dg.MultiSlotDataGenerator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        dg.MultiSlotDataGenerator.__init__(self)
        if os.path.isfile(config):
            with open(config, 'r') as rb:
                _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
        else:
            raise ValueError("reader config only support yaml")
        #envs.set_global_envs(_config)
        #envs.update_workspace()

    def init(self, sparse_slots, dense_slots, padding=0):
        from operator import mul
        self.sparse_slots = sparse_slots.strip().split(" ")
        self.dense_slots = dense_slots.strip().split(" ")
        self.dense_slots_shape = [
            reduce(mul,
                   [int(j) for j in i.split(":")[1].strip("[]").split(",")])
            for i in self.dense_slots
        ]
        self.dense_slots = [i.split(":")[0] for i in self.dense_slots]
        self.slots = self.dense_slots + self.sparse_slots
        self.slot2index = {}
        self.visit = {}
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding

    def generate_sample(self, l):
        def reader():
            line = l.strip().split(" ")
            output = [(i, []) for i in self.slots]
            for i in line:
                slot_feasign = i.split(":")
                slot = slot_feasign[0]
                if slot not in self.slots:
                    continue
                if slot in self.sparse_slots:
                    feasign = int(slot_feasign[1])
                else:
                    feasign = float(slot_feasign[1])
                output[self.slot2index[slot]][1].append(feasign)
                self.visit[slot] = True
            for i in self.visit:
                slot = i
                if not self.visit[slot]:
                    if i in self.dense_slots:
                        output[self.slot2index[i]][1].extend(
                            [self.padding] *
                            self.dense_slots_shape[self.slot2index[i]])
                    else:
                        output[self.slot2index[i]][1].extend([self.padding])
                else:
                    self.visit[slot] = False
            yield output

        return reader
