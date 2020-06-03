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

__all__ = ["StartupBase", "SingleStartup", "PSStartup", "CollectiveStartup"]


class StartupBase(object):
    """R
    """

    def __init__(self, context):
        pass

    def startup(self, context):
        pass

    def load(self, context, is_fleet=False):
        dirname = envs.get_global_env(
            "runner." + context["runner_name"] + ".init_model_path", None)
        if dirname is None or dirname == "":
            return
        print("going to load ", dirname)
        if is_fleet:
            context["fleet"].load_persistables(context["exe"], dirname)
        else:
            fluid.io.load_persistables(context["exe"], dirname)


class SingleStartup(StartupBase):
    def __init__(self, context):
        pass

    def startup(self, context):
        for model_dict in context["env"]["phase"]:
            with fluid.scope_guard(context["_model"][model_dict["name"]][2]):
                context["exe"].run(context["_model"][model_dict["name"]][1])
                train_prog = context["_model"][model_dict["name"]][0]
                startup_prog = context["_model"][model_dict["name"]][1]
                with fluid.program_guard(train_prog, startup_prog):
                    self.load(context)
        context["status"] = "train_pass"


class PSStartup(StartupBase):
    def __init__(self, context):
        pass

    def startup(self, context):
        model_dict = context["env"]["phase"][0]
        with fluid.scope_guard(context["_model"][model_dict["name"]][2]):
            context["exe"].run(context["_model"][model_dict["name"]][1])
            train_prog = context["_model"][model_dict["name"]][0]
            startup_prog = context["_model"][model_dict["name"]][1]
            with fluid.program_guard(train_prog, startup_prog):
                self.load(context, True)
        context["status"] = "train_pass"


class CollectiveStartup(StartupBase):
    def __init__(self, context):
        pass

    def startup(self, context):
        pass
