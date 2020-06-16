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

import os
import warnings

import paddle.fluid as fluid
from paddlerec.core.utils import envs
from paddlerec.core.trainers.framework.dataset import DataLoader, QueueDataset

__all__ = [
    "NetworkBase", "SingleNetwork", "PSNetwork", "PslibNetwork",
    "CollectiveNetwork"
]


class NetworkBase(object):
    """R
    """

    def __init__(self, context):
        pass

    def build_network(self, context):
        pass


class SingleNetwork(NetworkBase):
    """R
    """

    def __init__(self, context):
        print("Running SingleNetwork.")
        pass

    def build_network(self, context):
        context["model"] = {}
        for model_dict in context["phases"]:
            context["model"][model_dict["name"]] = {}
            train_program = fluid.Program()
            startup_program = fluid.Program()
            scope = fluid.Scope()
            dataset_name = model_dict["dataset_name"]

            with fluid.program_guard(train_program, startup_program):
                with fluid.unique_name.guard():
                    with fluid.scope_guard(scope):
                        model_path = envs.os_path_adapter(
                            envs.workspace_adapter(model_dict["model"]))
                        model = envs.lazy_instance_by_fliename(
                            model_path, "Model")(context["env"])

                        if context["is_infer"]:
                            model._infer_data_var = model.input_data(
                                is_infer=context["is_infer"],
                                dataset_name=model_dict["dataset_name"])
                        else:
                            model._data_var = model.input_data(
                                dataset_name=model_dict["dataset_name"])

                        if envs.get_global_env("dataset." + dataset_name +
                                               ".type") == "DataLoader":
                            model._init_dataloader(
                                is_infer=context["is_infer"])
                            data_loader = DataLoader(context)
                            data_loader.get_dataloader(context, dataset_name,
                                                       model._data_loader)

                        if context["is_infer"]:
                            model.net(model._infer_data_var,
                                      context["is_infer"])
                        else:
                            model.net(model._data_var, context["is_infer"])
                            optimizer = model.optimizer()
                            optimizer.minimize(model._cost)
            context["model"][model_dict["name"]][
                "main_program"] = train_program
            context["model"][model_dict["name"]][
                "startup_program"] = startup_program
            context["model"][model_dict["name"]]["scope"] = scope
            context["model"][model_dict["name"]]["model"] = model
            context["model"][model_dict["name"]][
                "default_main_program"] = train_program.clone()
            context["model"][model_dict["name"]]["compiled_program"] = None

        context["dataset"] = {}
        for dataset in context["env"]["dataset"]:
            type = envs.get_global_env("dataset." + dataset["name"] + ".type")
            if type != "DataLoader":
                dataset_class = QueueDataset(context)
                context["dataset"][dataset[
                    "name"]] = dataset_class.create_dataset(dataset["name"],
                                                            context)

        context["status"] = "startup_pass"


class PSNetwork(NetworkBase):
    def __init__(self, context):
        print("Running PSNetwork.")
        pass

    def build_network(self, context):
        context["model"] = {}
        if len(context["env"]["phase"]) > 1:
            warnings.warn(
                "Cluster Train Only Support One Phase.",
                category=UserWarning,
                stacklevel=2)
        model_dict = context["env"]["phase"][0]
        context["model"][model_dict["name"]] = {}
        dataset_name = model_dict["dataset_name"]

        model_path = envs.os_path_adapter(
            envs.workspace_adapter(model_dict["model"]))
        model = envs.lazy_instance_by_fliename(model_path,
                                               "Model")(context["env"])
        model._data_var = model.input_data(
            dataset_name=model_dict["dataset_name"])
        if envs.get_global_env("dataset." + dataset_name +
                               ".type") == "DataLoader":
            model._init_dataloader(is_infer=False)
            data_loader = DataLoader(context)
            data_loader.get_dataloader(context, dataset_name,
                                       model._data_loader)
        model.net(model._data_var, False)
        optimizer = model.optimizer()
        strategy = self._build_strategy(context)
        optimizer = context["fleet"].distributed_optimizer(optimizer, strategy)
        optimizer.minimize(model._cost)

        context["model"][model_dict["name"]]["main_program"] = context[
            "fleet"].main_program
        context["model"][model_dict["name"]]["startup_program"] = context[
            "fleet"].startup_program
        context["model"][model_dict["name"]]["scope"] = fluid.global_scope()
        context["model"][model_dict["name"]]["model"] = model
        context["model"][model_dict["name"]]["default_main_program"] = context[
            "fleet"].main_program.clone()
        context["model"][model_dict["name"]]["compiled_program"] = None

        if context["fleet"].is_server():
            self._server(context)
        else:
            context["fleet"].init_worker()
            context["dataset"] = {}
            for dataset in context["env"]["dataset"]:
                type = envs.get_global_env("dataset." + dataset["name"] +
                                           ".type")
                if type != "DataLoader":
                    dataset_class = QueueDataset(context)
                    context["dataset"][dataset[
                        "name"]] = dataset_class.create_dataset(
                            dataset["name"], context)
            context["status"] = "startup_pass"

    def _build_strategy(self, context):
        from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory
        mode = envs.get_runtime_environ("train.trainer.strategy")
        assert mode in ["async", "geo", "sync", "half_async"]

        strategy = None

        if mode == "async":
            strategy = StrategyFactory.create_async_strategy()
        elif mode == "geo":
            push_num = envs.get_global_env("train.strategy.mode.push_num", 100)
            strategy = StrategyFactory.create_geo_strategy(push_num)
        elif mode == "sync":
            strategy = StrategyFactory.create_sync_strategy()
        elif mode == "half_async":
            strategy = StrategyFactory.create_half_async_strategy()

        assert strategy is not None

        context["strategy"] = strategy
        return strategy

    def _server(self, context):
        init_model_path = envs.get_global_env(
            "runner." + context["runner_name"] + ".init_model_path",
            default_value="")
        context["fleet"].init_server(init_model_path)
        context["fleet"].run_server()
        context['status'] = "terminal_pass"


class PslibNetwork(NetworkBase):
    def __init__(self, context):
        print("Running PslibNetwork.")
        pass

    def build_network(self, context):
        context["model"] = {}
        if len(context["env"]["phase"]) > 1:
            warnings.warn(
                "Cluster Train Only Support One Phase.",
                category=UserWarning,
                stacklevel=2)
        model_dict = context["env"]["phase"][0]
        train_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        dataset_name = model_dict["dataset_name"]

        with fluid.program_guard(train_program, startup_program):
            with fluid.unique_name.guard():
                with fluid.scope_guard(scope):
                    context["model"][model_dict["name"]] = {}
                    model_path = envs.os_path_adapter(
                        envs.workspace_adapter(model_dict["model"]))
                    model = envs.lazy_instance_by_fliename(
                        model_path, "Model")(context["env"])
                    model._data_var = model.input_data(
                        dataset_name=model_dict["dataset_name"])
                    if envs.get_global_env("dataset." + dataset_name +
                                           ".type") == "DataLoader":
                        model._init_dataloader(is_infer=False)
                        data_loader = DataLoader(context)
                        data_loader.get_dataloader(context, dataset_name,
                                                   model._data_loader)
                    model.net(model._data_var, False)
                    optimizer = model.optimizer()

                    optimizer = context["fleet"].distributed_optimizer(
                        optimizer)
                    optimizer.minimize([model._cost], [fluid.global_scope()])

                    context["model"][model_dict["name"]][
                        "main_program"] = train_program
                    context["model"][model_dict["name"]][
                        "startup_program"] = startup_program
                    context["model"][model_dict["name"]]["scope"] = scope
                    context["model"][model_dict["name"]]["model"] = model
                    context["model"][model_dict["name"]][
                        "default_main_program"] = train_program.clone()
                    context["model"][model_dict["name"]][
                        "compile_program"] = None

        if context["fleet"].is_server():
            self._server(context)
        else:
            context["dataset"] = {}
            for dataset in context["env"]["dataset"]:
                type = envs.get_global_env("dataset." + dataset["name"] +
                                           ".type")
                if type != "DataLoader":
                    dataset_class = QueueDataset(context)
                    context["dataset"][dataset[
                        "name"]] = dataset_class.create_dataset(
                            dataset["name"], context)
            context["status"] = "startup_pass"

    def _server(self, context):
        context["fleet"].run_server()
        context['status'] = "terminal_pass"


class CollectiveNetwork(NetworkBase):
    def __init__(self, context):
        print("Running CollectiveNetwork.")
        pass

    def build_network(self, context):
        context["model"] = {}
        if len(context["env"]["phase"]) > 1:
            warnings.warn(
                "Cluster Train Only Support One Phase.",
                category=UserWarning,
                stacklevel=2)
        model_dict = context["env"]["phase"][0]
        context["model"][model_dict["name"]] = {}
        dataset_name = model_dict["dataset_name"]

        train_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        with fluid.program_guard(train_program, startup_program):
            with fluid.scope_guard(scope):
                model_path = envs.os_path_adapter(
                    envs.workspace_adapter(model_dict["model"]))

                model = envs.lazy_instance_by_fliename(model_path,
                                                       "Model")(context["env"])
                model._data_var = model.input_data(
                    dataset_name=model_dict["dataset_name"])
                if envs.get_global_env("dataset." + dataset_name +
                                       ".type") == "DataLoader":
                    model._init_dataloader(is_infer=False)
                    data_loader = DataLoader(context)
                    data_loader.get_dataloader(context, dataset_name,
                                               model._data_loader)
                model.net(model._data_var, False)
                optimizer = model.optimizer()
                strategy = self._build_strategy(context)
                optimizer = context["fleet"].distributed_optimizer(optimizer,
                                                                   strategy)
                optimizer.minimize(model._cost)

                context["model"][model_dict["name"]]["main_program"] = context[
                    "fleet"].main_program
                context["model"][model_dict["name"]][
                    "startup_program"] = startup_program
                context["model"][model_dict["name"]]["scope"] = scope
                context["model"][model_dict["name"]]["model"] = model
                context["model"][model_dict["name"]][
                    "default_main_program"] = train_program
                context["model"][model_dict["name"]]["compiled_program"] = None

        context["dataset"] = {}
        for dataset in context["env"]["dataset"]:
            type = envs.get_global_env("dataset." + dataset["name"] + ".type")
            if type != "DataLoader":
                dataset_class = QueueDataset(context)
                context["dataset"][dataset[
                    "name"]] = dataset_class.create_dataset(dataset["name"],
                                                            context)
        context["status"] = "startup_pass"

    def _build_strategy(self, context):
        from paddle.fluid.incubate.fleet.collective import DistributedStrategy
        exec_strategy = fluid.ExecutionStrategy()
        strategy = DistributedStrategy()
        strategy.exec_strategy = exec_strategy
        context["strategy"] = strategy
        return strategy
