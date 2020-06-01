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
from .dataset import DataLoader, QueueDataset

__all__ = ["NetWorkBase", "SingleNetWork", "PSNetwork", "CollectiveNetWork"]


class NetWorkBase(object):
    def __init__(self, context):
        pass

    def build_network(self, context):
        pass


class SingleNetWork(NetWorkBase):
    def __init__(self, context):
        pass

    def build_network(self, context):
        context["_model"] = {}
        for model_dict in context["env"]["phase"]:
            context["_model"][model_dict["name"]] = [None] * 5
            train_program = fluid.Program()
            startup_program = fluid.Program()
            scope = fluid.Scope()
            dataset_name = model_dict["dataset_name"]
            opt_name = envs.get_global_env("hyper_parameters.optimizer.class")
            opt_lr = envs.get_global_env(
                "hyper_parameters.optimizer.learning_rate")
            opt_strategy = envs.get_global_env(
                "hyper_parameters.optimizer.strategy")
            with fluid.program_guard(train_program, startup_program):
                with fluid.unique_name.guard():
                    with fluid.scope_guard(scope):
                        model_path = model_dict["model"].replace(
                            "{workspace}",
                            envs.path_adapter(context["env"]["workspace"]))
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
                            optimizer = model._build_optimizer(opt_name, opt_lr,
                                                               opt_strategy)
                            optimizer.minimize(model._cost)
            context["_model"][model_dict["name"]][0] = train_program
            context["_model"][model_dict["name"]][1] = startup_program
            context["_model"][model_dict["name"]][2] = scope
            context["_model"][model_dict["name"]][3] = model
            context["_model"][model_dict["name"]][4] = train_program.clone()

        context["dataset"] = {}
        for dataset in context["env"]["dataset"]:
            if dataset["type"] != "DataLoader":
                dataset_class = QueueDataset(context)
                context["dataset"][dataset["name"]] = dataset_class.create_dataset(dataset[
                    "name"], context)

        context["status"] = "startup_pass"


class PSNetwork(NetWorkBase):
    def __init__(self, context):
        pass

    def build_network(self, context):
        context["_model"] = {}
        if len(context["env"]["phase"]) > 1:
            warnings.warn("Cluster Train Only Support One Phase.",
                          category=UserWarning, stacklevel=2)
        model_dict = context["env"]["phase"][0]
        context["_model"][model_dict["name"]] = [None] * 5
        dataset_name = model_dict["dataset_name"]
        opt_name = envs.get_global_env("hyper_parameters.optimizer.class")
        opt_lr = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate")
        opt_strategy = envs.get_global_env(
            "hyper_parameters.optimizer.strategy")
        model_path = model_dict["model"].replace(
            "{workspace}",
            envs.path_adapter(context["env"]["workspace"]))
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
        optimizer = model._build_optimizer(opt_name, opt_lr,
                                           opt_strategy)
        strategy = self._build_strategy(context)
        optimizer = context["fleet"].distributed_optimizer(
            optimizer,  strategy)
        optimizer.minimize(model._cost)

        if opt_name.upper() == "SGD":
            os.environ["FLAGS_communicator_is_sgd_optimizer"] = '1'
        else:
            os.environ["FLAGS_communicator_is_sgd_optimizer"] = '0'

        context["_model"][model_dict["name"]
                          ][0] = context["fleet"].main_program
        context["_model"][model_dict["name"]
                          ][1] = context["fleet"].startup_program
        context["_model"][model_dict["name"]][2] = fluid.global_scope()
        context["_model"][model_dict["name"]][3] = model
        context["_model"][model_dict["name"]
                          ][4] = context["fleet"].main_program.clone()

        if context["fleet"].is_server():
            self._server(context)
        else:
            context["fleet"].init_worker()
            context["dataset"] = {}
            for dataset in context["env"]["dataset"]:
                if dataset["type"] != "DataLoader":
                    dataset_class = QueueDataset(context)
                    context["dataset"][dataset["name"]] = dataset_class.create_dataset(dataset[
                        "name"], context)
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
        init_model_path = envs.get_global_env("runner." + context["runner_name"] + ".init_model",
                                              default_value="")
        context["fleet"].init_server(init_model_path)
        context["fleet"].run_server()
        context['status'] = "terminal_pass"


class CollectiveNetWork(NetWorkBase):
    def __init__(self, context):
        pass

    def build_network(self, context):
        pass
