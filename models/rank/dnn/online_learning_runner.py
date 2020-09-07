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
import warnings
import numpy as np
import logging
import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.metric import Metric
from paddlerec.core.trainers.framework.runner import RunnerBase

logging.basicConfig(
    format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)


class OnlineLearningRunner(RunnerBase):
    def __init__(self, context):
        print("Running OnlineLearningRunner.")

    def run(self, context):
        epochs = int(
            envs.get_global_env("runner." + context["runner_name"] +
                                ".epochs"))
        model_dict = context["env"]["phase"][0]
        model_class = context["model"][model_dict["name"]]["model"]
        metrics = model_class._metrics

        dataset_list = []
        dataset_index = 0
        for day_index in range(len(days)):
            day = days[day_index]
            cur_path = "%s/%s" % (path, str(day))
            filelist = fleet.split_files(hdfs_ls([cur_path]))
            dataset = create_dataset(use_var, filelist)
            dataset_list.append(dataset)
            dataset_index += 1

        dataset_index = 0
        for epoch in range(len(days)):
            day = days[day_index]
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
