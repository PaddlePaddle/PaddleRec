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
import time
import numpy as np
from datetime import datetime
import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.metric import Metric

__all__ = [
    "RunnerBase", "SingleRunner", "PSRunner", "CollectiveRunner", "PslibRunner"
]


def as_numpy(tensor):
    """
    Convert a Tensor to a numpy.ndarray, its only support Tensor without LoD information.
    For higher dimensional sequence data, please use LoDTensor directly.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy

          new_scope = fluid.Scope()
          with fluid.scope_guard(new_scope):
              fluid.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), fluid.CPUPlace())
          tensor = new_scope.find_var("data").get_tensor()
          fluid.executor.as_numpy(tensor) # or numpy.array(new_scope.find_var("data").get_tensor())

    Args:
       tensor(Variable): a instance of Tensor

    Returns:
        numpy.ndarray
    """
    if isinstance(tensor, fluid.core.LoDTensorArray):
        return [as_numpy(t) for t in tensor]
    if isinstance(tensor, list):
        return [as_numpy(t) for t in tensor]
    assert isinstance(tensor, fluid.core.LoDTensor)
    lod = tensor.lod()
    # (todo) need print lod or return it for user
    if tensor._is_initialized():
        return np.array(tensor)
    else:
        return None


class RunnerBase(object):
    """R
    """

    def __init__(self, context):
        pass

    def exuctor(self, context):
        pass

    def _run(self, context, model_dict):
        reader_name = model_dict["dataset_name"]
        name = "dataset." + reader_name + "."

        if envs.get_global_env(name + "type") == "DataLoader":
            return self._executor_dataloader_train(model_dict, context)
        else:
            self._executor_dataset_train(model_dict, context)
            return None

    def _executor_dataset_train(self, model_dict, context):
        reader_name = model_dict["dataset_name"]
        model_name = model_dict["name"]
        model_class = context["model"][model_dict["name"]]["model"]

        fetch_vars = []
        fetch_alias = []
        fetch_period = int(
            envs.get_global_env("runner." + context["runner_name"] +
                                ".print_interval", 20))
        scope = context["model"][model_name]["scope"]
        program = context["model"][model_name]["main_program"]
        reader = context["dataset"][reader_name]

        with fluid.scope_guard(scope):
            if context["is_infer"]:
                metrics = model_class.get_infer_results()
                if metrics:
                    fetch_vars = metrics.values()
                    fetch_alias = metrics.keys()
                context["exe"].infer_from_dataset(
                    program=program,
                    dataset=reader,
                    fetch_list=fetch_vars,
                    fetch_info=fetch_alias,
                    print_period=fetch_period,
                    debug=envs.get_global_env("debug", False))
            else:
                metrics = model_class.get_metrics()
                if metrics:
                    fetch_vars = metrics.values()
                    fetch_alias = metrics.keys()
                with fluid.scope_guard(scope):
                    context["exe"].train_from_dataset(
                        program=program,
                        dataset=reader,
                        fetch_list=fetch_vars,
                        fetch_info=fetch_alias,
                        print_period=fetch_period,
                        debug=envs.get_global_env("debug", False))

    def _executor_dataloader_train(self, model_dict, context):
        model_name = model_dict["name"]
        model_class = context["model"][model_dict["name"]]["model"]
        program = self._get_dataloader_program(model_dict, context)

        fetch_period = int(
            envs.get_global_env("runner." + context["runner_name"] +
                                ".print_interval", 20))
        if context["is_infer"]:
            metrics = model_class.get_infer_results()
        else:
            metrics = model_class.get_metrics()

        metrics_varnames = []
        metrics_format = []

        if context["is_infer"]:
            metrics_format.append("{{}}\t[Infer]\t{}: {{}}".format("batch"))
        else:
            metrics_format.append("{{}}\t[Train]\t{}: {{}}".format("batch"))

        metrics_format.append("{}: {{:.2f}}s".format("time_each_interval"))

        metrics_names = ["total_batch"]
        for name, var in metrics.items():
            metrics_names.append(name)
            metrics_varnames.append(var.name)
            metrics_format.append("{}: {{}}".format(name))
        metrics_format = ", ".join(metrics_format)

        reader = context["model"][model_dict["name"]]["model"]._data_loader
        reader.start()
        batch_id = 0
        begin_time = time.time()
        scope = context["model"][model_name]["scope"]
        result = None
        with fluid.scope_guard(scope):
            try:
                while True:
                    metrics_tensors = context["exe"].run(
                        program=program,
                        fetch_list=metrics_varnames,
                        return_numpy=False)

                    if batch_id % fetch_period == 0 and batch_id != 0:
                        metrics = [
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        ]
                        metrics.extend([batch_id])
                        end_time = time.time()
                        seconds = end_time - begin_time
                        metrics.extend([seconds])
                        begin_time = end_time

                        metrics_rets = [
                            as_numpy(metrics_tensor)
                            for metrics_tensor in metrics_tensors
                        ]
                        metrics.extend(metrics_rets)

                        print(metrics_format.format(*metrics))
                    batch_id += 1
            except fluid.core.EOFException:
                reader.reset()

        if batch_id > 0:
            result = dict(zip(metrics_names, metrics))
        return result

    def _get_dataloader_program(self, model_dict, context):
        model_name = model_dict["name"]
        if context["model"][model_name]["compiled_program"] == None:
            if context["is_infer"]:
                program = context["model"][model_name]["main_program"]
            elif context["is_fleet"]:
                if context["fleet_mode"].upper() == "PS":
                    program = self._get_ps_program(model_dict, context)
                elif context["fleet_mode"].upper() == "COLLECTIVE":
                    program = context["model"][model_name]["main_program"]
            elif not context["is_fleet"]:
                if context["device"].upper() == "CPU":
                    program = self._get_single_cpu_program(model_dict, context)
                elif context["device"].upper() == "GPU":
                    program = self._get_single_gpu_program(model_dict, context)
            context["model"][model_name]["compiled_program"] = program
        return context["model"][model_name]["compiled_program"]

    def _get_strategy(self, model_dict, context):
        _build_strategy = fluid.BuildStrategy()
        _exe_strategy = fluid.ExecutionStrategy()

        # 0: kCoeffNumDevice; 1: One; 2: Customized
        _gradient_scale_strategy = model_dict.get("gradient_scale_strategy", 0)
        if _gradient_scale_strategy == 0:
            gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.CoeffNumDevice
        elif _gradient_scale_strategy == 1:
            gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.One
        elif _gradient_scale_strategy == 2:
            gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.Customized
        else:
            raise ValueError(
                "Unsupported config. gradient_scale_strategy must be one of [0, 1, 2]."
            )
        _build_strategy.gradient_scale_strategy = gradient_scale_strategy

        if "thread_num" in model_dict and model_dict["thread_num"] > 1:
            _build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
            _exe_strategy.num_threads = model_dict["thread_num"]
            os.environ['CPU_NUM'] = str(_exe_strategy.num_threads)

        return _exe_strategy, _build_strategy

    def _get_single_gpu_program(self, model_dict, context):
        model_name = model_dict["name"]
        return context["model"][model_name]["main_program"].clone()

    def _get_single_cpu_program(self, model_dict, context):
        model_name = model_dict["name"]
        model_class = context["model"][model_dict["name"]]["model"]
        program = context["model"][model_name]["main_program"].clone()
        _exe_strategy, _build_strategy = self._get_strategy(model_dict,
                                                            context)

        program = fluid.compiler.CompiledProgram(program).with_data_parallel(
            loss_name=model_class.get_avg_cost().name,
            build_strategy=_build_strategy,
            exec_strategy=_exe_strategy)
        return program

    def _get_ps_program(self, model_dict, context):
        model_name = model_dict["name"]
        model_class = context["model"][model_dict["name"]]["model"]
        program = context["model"][model_name]["main_program"].clone()

        _build_strategy = context["strategy"].get_build_strategy()
        _exe_strategy = context["strategy"].get_execute_strategy()

        if "thread_num" in model_dict and model_dict["thread_num"] > 1:
            _build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
            _exe_strategy.num_threads = model_dict["thread_num"]
            os.environ['CPU_NUM'] = str(_exe_strategy.num_threads)

        _gradient_scale_strategy = model_dict.get("gradient_scale_strategy", 0)
        if _gradient_scale_strategy == 0:
            gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.CoeffNumDevice
        elif _gradient_scale_strategy == 1:
            gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.One
        elif _gradient_scale_strategy == 2:
            gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.Customized
        else:
            raise ValueError(
                "Unsurpported config. gradient_scale_strategy must be one of [0, 1, 2]."
            )
        _build_strategy.gradient_scale_strategy = gradient_scale_strategy

        program = fluid.compiler.CompiledProgram(program).with_data_parallel(
            loss_name=model_class.get_avg_cost().name,
            build_strategy=_build_strategy,
            exec_strategy=_exe_strategy)
        return program

    def save(self, epoch_id, context, is_fleet=False):
        def need_save(epoch_id, epoch_interval, is_last=False):
            name = "runner." + context["runner_name"] + "."
            total_epoch = int(envs.get_global_env(name + "epochs", 1))
            if epoch_id + 1 == total_epoch:
                is_last = True

            if is_last:
                return True
            if epoch_id == -1:
                return False

            return (epoch_id + 1) % epoch_interval == 0

        def save_inference_model():
            name = "runner." + context["runner_name"] + "."
            save_interval = int(
                envs.get_global_env(name + "save_inference_interval", -1))
            if not need_save(epoch_id, save_interval, False):
                return
            feed_varnames = envs.get_global_env(
                name + "save_inference_feed_varnames", [])
            fetch_varnames = envs.get_global_env(
                name + "save_inference_fetch_varnames", [])
            if feed_varnames is None or fetch_varnames is None or feed_varnames == "" or fetch_varnames == "" or \
                    len(feed_varnames) == 0 or len(fetch_varnames) == 0:
                return
            fetch_vars = [
                fluid.default_main_program().global_block().vars[varname]
                for varname in fetch_varnames
            ]
            dirname = envs.get_global_env(name + "save_inference_path", None)

            assert dirname is not None
            dirname = os.path.join(dirname, str(epoch_id))

            if is_fleet:
                context["fleet"].save_inference_model(
                    context["exe"], dirname, feed_varnames, fetch_vars)
            else:
                fluid.io.save_inference_model(dirname, feed_varnames,
                                              fetch_vars, context["exe"])

        def save_persistables():
            name = "runner." + context["runner_name"] + "."
            save_interval = int(
                envs.get_global_env(name + "save_checkpoint_interval", -1))
            if not need_save(epoch_id, save_interval, False):
                return
            dirname = envs.get_global_env(name + "save_checkpoint_path", None)
            if dirname is None or dirname == "":
                return
            dirname = os.path.join(dirname, str(epoch_id))
            if is_fleet:
                context["fleet"].save_persistables(context["exe"], dirname)
            else:
                fluid.io.save_persistables(context["exe"], dirname)

        save_persistables()
        save_inference_model()


class SingleRunner(RunnerBase):
    """R
    """

    def __init__(self, context):
        print("Running SingleRunner.")
        pass

    def run(self, context):
        epochs = int(
            envs.get_global_env("runner." + context["runner_name"] +
                                ".epochs"))
        for epoch in range(epochs):
            for model_dict in context["phases"]:
                model_class = context["model"][model_dict["name"]]["model"]
                metrics = model_class._metrics

                begin_time = time.time()
                result = self._run(context, model_dict)
                end_time = time.time()
                seconds = end_time - begin_time
                message = "epoch {} done, use time: {}".format(epoch, seconds)
                metrics_result = []
                for key in metrics:
                    if isinstance(metrics[key], Metric):
                        _str = metrics[key].calc_global_metrics(
                            None,
                            context["model"][model_dict["name"]]["scope"])
                        metrics_result.append(_str)
                    elif result is not None:
                        _str = "{}={}".format(key, result[key])
                        metrics_result.append(_str)
                if len(metrics_result) > 0:
                    message += ", global metrics: " + ", ".join(metrics_result)
                print(message)

                with fluid.scope_guard(context["model"][model_dict["name"]][
                        "scope"]):
                    train_prog = context["model"][model_dict["name"]][
                        "default_main_program"]
                    startup_prog = context["model"][model_dict["name"]][
                        "startup_program"]
                    with fluid.program_guard(train_prog, startup_prog):
                        self.save(epoch, context)
        context["status"] = "terminal_pass"


class PSRunner(RunnerBase):
    def __init__(self, context):
        print("Running PSRunner.")
        pass

    def run(self, context):
        epochs = int(
            envs.get_global_env("runner." + context["runner_name"] +
                                ".epochs"))
        model_dict = context["env"]["phase"][0]
        model_class = context["model"][model_dict["name"]]["model"]
        metrics = model_class._metrics
        for epoch in range(epochs):
            begin_time = time.time()
            result = self._run(context, model_dict)
            end_time = time.time()
            seconds = end_time - begin_time
            message = "epoch {} done, use time: {}".format(epoch, seconds)

            # TODO, wait for PaddleCloudRoleMaker supports gloo
            from paddle.fluid.incubate.fleet.base.role_maker import GeneralRoleMaker
            if context["fleet"] is not None and isinstance(context["fleet"],
                                                           GeneralRoleMaker):
                metrics_result = []
                for key in metrics:
                    if isinstance(metrics[key], Metric):
                        _str = metrics[key].calc_global_metrics(
                            context["fleet"],
                            context["model"][model_dict["name"]]["scope"])
                        metrics_result.append(_str)
                    elif result is not None:
                        _str = "{}={}".format(key, result[key])
                        metrics_result.append(_str)
                if len(metrics_result) > 0:
                    message += ", global metrics: " + ", ".join(metrics_result)
            print(message)
            with fluid.scope_guard(context["model"][model_dict["name"]][
                    "scope"]):
                train_prog = context["model"][model_dict["name"]][
                    "main_program"]
                startup_prog = context["model"][model_dict["name"]][
                    "startup_program"]
                with fluid.program_guard(train_prog, startup_prog):
                    self.save(epoch, context, True)
        context["status"] = "terminal_pass"


class CollectiveRunner(RunnerBase):
    def __init__(self, context):
        print("Running CollectiveRunner.")
        pass

    def run(self, context):
        epochs = int(
            envs.get_global_env("runner." + context["runner_name"] +
                                ".epochs"))
        model_dict = context["env"]["phase"][0]
        for epoch in range(epochs):
            begin_time = time.time()
            self._run(context, model_dict)
            end_time = time.time()
            seconds = end_time - begin_time
            print("epoch {} done, use time: {}".format(epoch, seconds))
            with fluid.scope_guard(context["model"][model_dict["name"]][
                    "scope"]):
                train_prog = context["model"][model_dict["name"]][
                    "default_main_program"]
                startup_prog = context["model"][model_dict["name"]][
                    "startup_program"]
                with fluid.program_guard(train_prog, startup_prog):
                    self.save(epoch, context, True)
        context["status"] = "terminal_pass"


class PslibRunner(RunnerBase):
    def __init__(self, context):
        print("Running PSRunner.")
        pass

    def run(self, context):
        context["fleet"].init_worker()
        model_dict = context["env"]["phase"][0]
        epochs = int(
            envs.get_global_env("runner." + context["runner_name"] +
                                ".epochs"))
        for epoch in range(epochs):
            begin_time = time.time()
            self._run(context, model_dict)
            end_time = time.time()
            seconds = end_time - begin_time
            print("epoch {} done, use time: {}".format(epoch, seconds))
        """
        # online Training Can do more, As shown below:

        begin_day = datetime.datetime.strptime("begin_day_d", '%Y%m%d')
        days = int(
            envs.get_global_env("runner." + context["runner_name"] + ".days"))
        for day in range(days):
            for hour in range(24):
                day = begin_day + datetime.timedelta(days=day, hours=hour)
                day_s = day.strftime('%Y%m%d/%H')

                for dataset in envs.get_global_env("dataset"):
                    if dataset["type"] != "DataLoader":
                        name = dataset["name"]
                        train_data_path = envs.get_global_env(name +
                                                              "data_path")
                        train_data_path = os.path.join(train_data_path, day_s)

                        file_list = [
                            os.path.join(train_data_path, x)
                            for x in os.listdir(train_data_path)
                        ]
                        context["dataset"][name].set_filelist(file_list)

                for epoch in range(epochs):
                    begin_time = time.time()
                    self._run(context, model_dict)
                    end_time = time.time()
                    seconds = end_time - begin_time
                    print("epoch {} done, use time: {}".format(epoch, seconds))
                    with fluid.scope_guard(context["model"][model_dict["name"]]
                                           ["scope"]):
                        train_prog = context["model"][model_dict["name"]][
                            "default_main_program"]
                        startup_prog = context["model"][model_dict["name"]][
                            "startup_program"]
                        with fluid.program_guard(train_prog, startup_prog):
                            self.save(epoch, context, True)

        """
        context["status"] = "terminal_pass"


class SingleInferRunner(RunnerBase):
    def __init__(self, context):
        print("Running SingleInferRunner.")
        pass

    def run(self, context):
        self._dir_check(context)

        self.epoch_model_name_list.sort()
        for index, epoch_name in enumerate(self.epoch_model_name_list):
            for model_dict in context["phases"]:
                model_class = context["model"][model_dict["name"]]["model"]
                metrics = model_class._infer_results
                self._load(context, model_dict,
                           self.epoch_model_path_list[index])
                begin_time = time.time()
                result = self._run(context, model_dict)
                end_time = time.time()
                seconds = end_time - begin_time
                message = "Infer {} of epoch {} done, use time: {}".format(
                    model_dict["name"], epoch_name, seconds)
                metrics_result = []
                for key in metrics:
                    if isinstance(metrics[key], Metric):
                        _str = metrics[key].calc_global_metrics(
                            None,
                            context["model"][model_dict["name"]]["scope"])
                        metrics_result.append(_str)
                    elif result is not None:
                        _str = "{}={}".format(key, result[key])
                        metrics_result.append(_str)
                if len(metrics_result) > 0:
                    message += ", global metrics: " + ", ".join(metrics_result)
                print(message)

        context["status"] = "terminal_pass"

    def _load(self, context, model_dict, model_path):
        if model_path is None or model_path == "":
            return
        print("load persistables from", model_path)

        with fluid.scope_guard(context["model"][model_dict["name"]]["scope"]):
            train_prog = context["model"][model_dict["name"]]["main_program"]
            startup_prog = context["model"][model_dict["name"]][
                "startup_program"]
            with fluid.program_guard(train_prog, startup_prog):
                fluid.io.load_persistables(
                    context["exe"], model_path, main_program=train_prog)
            clear_metrics = context["model"][model_dict["name"]][
                "model"].get_clear_metrics()
            for var in clear_metrics:
                var.clear()

    def _dir_check(self, context):
        dirname = envs.get_global_env(
            "runner." + context["runner_name"] + ".init_model_path", None)
        self.epoch_model_path_list = []
        self.epoch_model_name_list = []

        for file in os.listdir(dirname):
            file_path = os.path.join(dirname, file)
            if os.path.isdir(file_path):
                self.epoch_model_path_list.append(file_path)
                self.epoch_model_name_list.append(file)

        if len(self.epoch_model_path_list) == 0:
            self.epoch_model_path_list.append(dirname)
            self.epoch_model_name_list.append(dirname)
