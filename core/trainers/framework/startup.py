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

    def load(self, context, is_fleet=False, main_program=None):
        dirname = envs.get_global_env(
            "runner." + context["runner_name"] + ".init_model_path", None)
        if dirname is None or dirname == "":
            return
        print("going to load ", dirname)
        if is_fleet:
            context["fleet"].load_persistables(context["exe"], dirname)
        else:
            fluid.io.load_persistables(
                context["exe"], dirname, main_program=main_program)


class SingleStartup(StartupBase):
    """R
    """

    def __init__(self, context):
        print("Running SingleStartup.")
        pass

    def startup(self, context):
        for model_dict in context["phases"]:
            with fluid.scope_guard(context["model"][model_dict["name"]][
                    "scope"]):
                train_prog = context["model"][model_dict["name"]][
                    "main_program"]
                startup_prog = context["model"][model_dict["name"]][
                    "startup_program"]
                with fluid.program_guard(train_prog, startup_prog):
                    context["exe"].run(startup_prog)
                    self.load(context, main_program=train_prog)
        context["status"] = "train_pass"


class PSStartup(StartupBase):
    def __init__(self, context):
        print("Running PSStartup.")
        pass

    def startup(self, context):
        model_dict = envs.get_global_env("phase")[0]
        with fluid.scope_guard(context["model"][model_dict["name"]]["scope"]):

            train_prog = context["model"][model_dict["name"]]["main_program"]
            startup_prog = context["model"][model_dict["name"]][
                "startup_program"]
            with fluid.program_guard(train_prog, startup_prog):
                context["exe"].run(startup_prog)
                self.load(context, True)
        context["status"] = "train_pass"


class CollectiveStartup(StartupBase):
    def __init__(self, context):
        print("Running CollectiveStartup.")
        pass

    def startup(self, context):
        model_dict = envs.get_global_env("phase")[0]
        with fluid.scope_guard(context["model"][model_dict["name"]]["scope"]):
            train_prog = context["model"][model_dict["name"]][
                "default_main_program"]
            startup_prog = context["model"][model_dict["name"]][
                "startup_program"]
            with fluid.program_guard(train_prog, startup_prog):
                context["exe"].run(startup_prog)
                self.load(context, True)
        context["status"] = "train_pass"
