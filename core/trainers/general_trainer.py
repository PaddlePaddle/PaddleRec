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
General Trainer, applicable to many situations: Single/Cluster/Local_Cluster + PS/COLLECTIVE
"""
from __future__ import print_function

import os

from paddlerec.core.utils import envs
from paddlerec.core.trainer import Trainer, EngineMode, FleetMode
import paddle


class GeneralTrainer(Trainer):
    """
    Trainer for various situations.
    """

    def __init__(self, config=None):
        paddle.enable_static()
        Trainer.__init__(self, config)
        self.processor_register()
        self.abs_dir = os.path.dirname(os.path.abspath(__file__))
        self.runner_env_name = "runner." + self._context["runner_name"]

    def processor_register(self):
        print("processor_register begin")
        self.regist_context_processor('uninit', self.instance)
        self.regist_context_processor('network_pass', self.network)
        self.regist_context_processor('startup_pass', self.startup)
        self.regist_context_processor('train_pass', self.runner)
        self.regist_context_processor('terminal_pass', self.terminal)

    def instance(self, context):
        instance_class_path = envs.get_global_env(
            self.runner_env_name + ".instance_class_path", default_value=None)
        if instance_class_path:
            instance_class = envs.lazy_instance_by_fliename(
                instance_class_path, "Instance")(context)
        else:
            if self.engine == EngineMode.SINGLE:
                instance_class_name = "SingleInstance"
            elif self.fleet_mode == FleetMode.PSLIB:
                instance_class_name = "PslibInstance"
            elif self.fleet_mode == FleetMode.PS:
                instance_class_name = "PSInstance"
            elif self.fleet_mode == FleetMode.COLLECTIVE:
                instance_class_name = "CollectiveInstance"
            else:
                raise ValueError("Instance Init Error")
            instance_path = os.path.join(self.abs_dir, "framework",
                                         "instance.py")
            instance_class = envs.lazy_instance_by_fliename(
                instance_path, instance_class_name)(context)

        instance_class.instance(context)

    def network(self, context):
        network_class_path = envs.get_global_env(
            self.runner_env_name + ".network_class_path", default_value=None)
        if network_class_path:
            network_class = envs.lazy_instance_by_fliename(network_class_path,
                                                           "Network")(context)
        else:
            if self.engine == EngineMode.SINGLE:
                network_class_name = "SingleNetwork"
            elif self.fleet_mode == FleetMode.PSLIB:
                network_class_name = "PslibNetwork"
            elif self.fleet_mode == FleetMode.PS:
                network_class_name = "PSNetwork"
            elif self.fleet_mode == FleetMode.COLLECTIVE:
                network_class_name = "CollectiveNetwork"
            else:
                raise ValueError("NetWork Init Error")
            network_path = os.path.join(self.abs_dir, "framework",
                                        "network.py")
            network_class = envs.lazy_instance_by_fliename(
                network_path, network_class_name)(context)

        network_class.build_network(context)

    def startup(self, context):
        startup_class_path = envs.get_global_env(
            self.runner_env_name + ".startup_class_path", default_value=None)
        if startup_class_path:
            startup_class = envs.lazy_instance_by_fliename(startup_class_path,
                                                           "Startup")(context)
        else:
            if self.engine == EngineMode.SINGLE and context["is_infer"]:
                startup_class_name = "SingleInferStartup"
            elif self.engine == EngineMode.SINGLE and not context["is_infer"]:
                startup_class_name = "SingleStartup"
            elif self.fleet_mode == FleetMode.PS or self.fleet_mode == FleetMode.PSLIB:
                startup_class_name = "PSStartup"
            elif self.fleet_mode == FleetMode.COLLECTIVE:
                startup_class_name = "CollectiveStartup"
            else:
                raise ValueError("Startup Init Error")
            startup_path = os.path.join(self.abs_dir, "framework",
                                        "startup.py")
            startup_class = envs.lazy_instance_by_fliename(
                startup_path, startup_class_name)(context)
        startup_class.startup(context)

    def runner(self, context):
        runner_class_path = envs.get_global_env(
            self.runner_env_name + ".runner_class_path", default_value=None)
        if runner_class_path:
            runner_class = envs.lazy_instance_by_fliename(runner_class_path,
                                                          "Runner")(context)
        else:
            if self.engine == EngineMode.SINGLE and context["is_infer"]:
                runner_class_name = "SingleInferRunner"
            elif self.engine == EngineMode.SINGLE and not context["is_infer"]:
                runner_class_name = "SingleRunner"
            elif self.fleet_mode == FleetMode.PSLIB:
                runner_class_name = "PslibRunner"
            elif self.fleet_mode == FleetMode.PS:
                runner_class_name = "PSRunner"
            elif self.fleet_mode == FleetMode.COLLECTIVE:
                runner_class_name = "CollectiveRunner"
            else:
                raise ValueError("Runner Init Error")
            runner_path = os.path.join(self.abs_dir, "framework", "runner.py")
            runner_class = envs.lazy_instance_by_fliename(
                runner_path, runner_class_name)(context)
        runner_class.run(context)

    def terminal(self, context):
        terminal_class_path = envs.get_global_env(
            self.runner_env_name + ".terminal_class_path", default_value=None)
        if terminal_class_path:
            terminal_class = envs.lazy_instance_by_fliename(
                terminal_class_path, "Terminal")(context)
            terminal_class.terminal(context)
        else:
            terminal_class_name = "TerminalBase"
            if self.engine != EngineMode.SINGLE and self.fleet_mode != FleetMode.COLLECTIVE:
                terminal_class_name = "PSTerminal"

            terminal_path = os.path.join(self.abs_dir, "framework",
                                         "terminal.py")
            terminal_class = envs.lazy_instance_by_fliename(
                terminal_path, terminal_class_name)(context)
        terminal_class.terminal(context)
        context['is_exit'] = True
