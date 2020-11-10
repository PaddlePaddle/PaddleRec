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

from __future__ import print_function

import warnings

import paddle.fluid as fluid
from paddlerec.core.utils import envs

__all__ = [
    "InstanceBase", "SingleInstance", "PSInstance", "PslibInstance",
    "CollectiveInstance"
]


class InstanceBase(object):
    """R
    """

    def __init__(self, context):
        pass

    def instance(self, context):
        pass


class SingleInstance(InstanceBase):
    def __init__(self, context):
        print("Running SingleInstance.")
        pass

    def instance(self, context):
        context['status'] = 'network_pass'


class FleetInstance(InstanceBase):
    def __init__(self, context):
        print("Running FleetInstance")

    def instance(self, context):
        import paddle.distributed.fleet.base.role_maker as role_maker
        import paddle.distributed.fleet as fleet
        is_collective = (context["fleet_mode"] == 'COLLECTIVE')
        role = role_maker.PaddleCloudRoleMaker(is_collective=is_collective)
        fleet.init(role)
        context['fleet'] = fleet
        context['role'] = role
        context['status'] = 'network_pass'
