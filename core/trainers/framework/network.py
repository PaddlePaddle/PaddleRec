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
import paddle
import paddle.fluid as fluid
from paddlerec.core.utils import envs
from paddlerec.core.trainers.framework.dataset import DataLoader, QueueDataset

__all__ = [
    "NetworkBase", "SingleNetwork", "PSNetwork", "PslibNetwork",
    "CollectiveNetwork", "FineTuningNetwork"
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
            train_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            scope = paddle.static.Scope()
            dataset_name = model_dict["dataset_name"]

            with paddle.static.program_guard(train_program, startup_program):
                with fluid.unique_name.guard():
                    with paddle.static.scope_guard(scope):
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

            if type == "QueueDataset":
                dataset_class = QueueDataset(context)
                context["dataset"][dataset[
                    "name"]] = dataset_class.create_dataset(dataset["name"],
                                                            context)

        context["status"] = "startup_pass"


class FineTuningNetwork(NetworkBase):
    """R
    """

    def __init__(self, context):
        print("Running FineTuningNetwork.")

    def build_network(self, context):
        context["model"] = {}
        for model_dict in context["phases"]:
            context["model"][model_dict["name"]] = {}
            train_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            scope = fluid.Scope()
            dataset_name = model_dict["dataset_name"]

            with paddle.static.program_guard(train_program, startup_program):
                with fluid.unique_name.guard():
                    with paddle.static.scope_guard(scope):
                        model_path = envs.os_path_adapter(
                            envs.workspace_adapter(model_dict["model"]))
                        model = envs.lazy_instance_by_fliename(
                            model_path, "Model")(context["env"])

                        model._data_var = model.input_data(
                            dataset_name=model_dict["dataset_name"])

                        if envs.get_global_env("dataset." + dataset_name +
                                               ".type") == "DataLoader":
                            model._init_dataloader(
                                is_infer=context["is_infer"])
                            data_loader = DataLoader(context)
                            data_loader.get_dataloader(context, dataset_name,
                                                       model._data_loader)

                        model.net(model._data_var, context["is_infer"])

                        finetuning_varnames = envs.get_global_env(
                            "runner." + context["runner_name"] +
                            ".finetuning_aspect_varnames",
                            default_value=[])

                        if len(finetuning_varnames) == 0:
                            raise ValueError(
                                "nothing need to be fine tuning, you may use other traning mode"
                            )

                        if len(finetuning_varnames) != 1:
                            raise ValueError(
                                "fine tuning mode can only accept one varname now"
                            )

                        varname = finetuning_varnames[0]
                        finetuning_vars = train_program.global_block().vars[
                            varname]
                        finetuning_vars.stop_gradient = True
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

            if type == "QueueDataset":
                dataset_class = QueueDataset(context)
                context["dataset"][dataset[
                    "name"]] = dataset_class.create_dataset(dataset["name"],
                                                            context)

        context["status"] = "startup_pass"


class FleetNetwork(NetworkBase):
    def __init__(self, context):
        print("Running FleetNetwork.")

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

        model.net(model._data_var, False)
        optimizer = model.optimizer()
        strategy = self._build_strategy(context)
        optimizer = context["fleet"].distributed_optimizer(optimizer, strategy)
        optimizer.minimize(model._cost)

        context["model"][model_dict["name"]][
            "main_program"] = paddle.static.default_main_program()
        context["model"][model_dict["name"]][
            "startup_program"] = paddle.static.default_startup_program()
        context["model"][model_dict["name"]][
            "scope"] = paddle.static.global_scope()
        context["model"][model_dict["name"]]["model"] = model
        context["model"][model_dict["name"]][
            "default_main_program"] = paddle.static.default_main_program(
            ).clone()
        context["model"][model_dict["name"]][
            "compiled_program"] = paddle.static.default_main_program()

        if context["fleet"].is_server():
            self._server(context)
        else:
            context["dataset"] = {}
            for phase in context["env"]["phase"]:
                type = envs.get_global_env("dataset." + phase["dataset_name"] +
                                           ".type")
                if type == "DataLoader":
                    data_loader = DataLoader(context)
                    data_loader.get_dataloader(context, dataset_name,
                                               model._data_loader)
                elif type == "QueueDataset":
                    if context["fleet_mode"] == "COLLECTIVE":
                        raise ValueError(
                            "Collective don't support QueueDataset training, please use DataLoader."
                        )
                    dataset_class = QueueDataset(context)
                    context["dataset"][phase[
                        "dataset_name"]] = dataset_class.create_dataset(
                            phase["dataset_name"], context)
            context["status"] = "startup_pass"

    def _build_strategy(self, context):
        from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory
        mode = envs.get_runtime_environ("train.trainer.strategy")
        assert mode in ["async", "geo", "sync"]

        strategy = None

        if context['fleet_mode'] == "PS":
            strategy = paddle.distributed.fleet.DistributedStrategy()
            if mode == 'async':
                strategy.a_sync = True
            elif mode == 'sync':
                strategy.a_sync = False
            elif mode == 'geo':
                strategy.a_sync = True
                strategy.a_sync_configs = {"k_steps": 400}
        elif context['fleet_mode'] == "COLLECTIVE":
            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.sync_nccl_allreduce = True
            strategy.nccl_comm_num = 2
            strategy.fuse_all_reduce_ops = True

            # build strategy
            build_strategy = fluid.BuildStrategy()
            build_strategy.enable_sequential_execution = True
            build_strategy.fuse_elewise_add_act_ops = True
            build_strategy.fuse_bn_act_ops = True
            build_strategy.enable_auto_fusion = True
            build_strategy.fuse_all_optimizer_ops = True
            strategy.build_strategy = build_strategy

            # execute strategy
            execution_strategy = paddle.static.ExecutionStrategy()
            execution_strategy.num_threads = int(os.getenv('CPU_NUM', 2))
            execution_strategy.num_iteration_per_drop_scope = 100
            execution_strategy.num_iteration_per_run = 1
            strategy.execution_strategy = execution_strategy

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
