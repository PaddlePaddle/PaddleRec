# -*- coding=utf-8 -*-
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

import time
import logging
import os
import numpy as np
import json
import paddle
import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.trainers.framework.runner import SingleInferRunner
from paddlerec.core.trainers.framework.terminal import TerminalBase
from paddlerec.core.trainer import EngineMode
from paddlerec.core.utils.util import shuffle_files
from paddlerec.core.metric import Metric

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


class Terminal(TerminalBase):
    def __init__(self, context):
        print("Running CTR-DNN Terminal")
        pass

    def get_example_num(self, file_list):
        count = 0
        for f in file_list:
            last_count = count
            for index, line in enumerate(open(f, 'r')):
                count += 1
            logger.info("file: %s has %s examples" % (f, count - last_count))
        logger.info("Total example: %s" % count)
        return count

    def terminal(self, context):
        work_path = "./benchmark_logs/"
        if not os.path.isdir(work_path):
            os.makedirs(work_path)

        train_files_path = "./train_data"
        file_list = [
            train_files_path + "/%s" % x for x in os.listdir(train_files_path)
        ]
        train_examples = self.get_example_num(file_list)
        result = {}

        if context['is_infer']:
            # 1. Save infer result
            auc_result = []
            for epoch in sorted(context['infer_result']):
                auc_result.append(context['infer_result'][epoch]['AUC'])
            result['auc'] = max(auc_result)
            result['auc_list'] = auc_result
            file_path = "./benchmark_logs/infer_log"
            with open(file_path, 'w') as f:
                f.writelines(str(result))
                logger.info("write infer log to work path:wirte %s" %
                            (file_path))
            logger.info("Infer log: {}".format(result))
        elif context['fleet_mode'] == "PS":
            # 2. Save training speed
            model_dict = context["env"]["phase"][0]
            running_time = context["model"][model_dict["name"]]["running_time"]
            speed = []
            for time in running_time:
                speed.append(train_examples / float(time))

            result['performance'] = np.mean(speed)
            result['performance_list'] = speed

            result['cost_time'] = np.mean(running_time)
            result['cost_time_list'] = running_time

            result['file_list'] = file_list
            result['examples'] = train_examples

            file_path = "./benchmark_logs/training_log"
            with open(file_path, 'w') as f:
                f.writelines(str(result))
                logger.info("write training log to work path:wirte %s" %
                            (file_path))
            logger.info("Trainer: {} Training log: {}".format(context[
                'fleet'].worker_index(), result))

        # 3. Stop Worker
        if context['fleet_mode'] == 'PS':
            context['fleet'].stop_worker()


class Runner(SingleInferRunner):
    def __init__(self, context):
        print("Running CTR-DNN SingleInferRunner.")
        pass

    def run(self, context):
        role = os.getenv("TRAINING_ROLE", "TRAINER")
        if role != 'TRAINER':
            context["status"] = "terminal_pass"
            return

        worker_id = int(os.getenv("PADDLE_TRAINER_ID", '0'))
        if worker_id != 0:
            context["status"] = "terminal_pass"
            return

        self._dir_check(context)
        context['infer_result'] = {}
        for index, epoch_name in enumerate(self.epoch_model_name_list):
            for model_dict in context["phases"]:
                model_class = context["model"][model_dict["name"]]["model"]
                context["model"][model_dict["name"]]["running_time"] = []
                metrics = model_class._infer_results
                self._load(context, model_dict,
                           self.epoch_model_path_list[index])

                begin_time = time.time()
                result = self._run(context, model_dict)
                end_time = time.time()
                seconds = end_time - begin_time
                message = "Infer {} of epoch {} done, use time: {}".format(
                    model_dict["name"], epoch_name, seconds)
                context["model"][model_dict["name"]]["running_time"].append(
                    seconds)
                metrics_result = []
                context['infer_result'][epoch_name] = {}
                for key in metrics:
                    if isinstance(metrics[key], Metric):
                        _str = metrics[key].calc_global_metrics(
                            None,
                            context["model"][model_dict["name"]]["scope"])
                        metrics_result.append(_str)
                    elif result is not None:
                        _str = "{}={}".format(key, result[key])
                        metrics_result.append(_str)
                        # for batch auc & auc
                        context['infer_result'][epoch_name][key] = result[key]
                if len(metrics_result) > 0:
                    message += ", final metrics: " + ", ".join(metrics_result)

                logging.info(message)
        context["status"] = "terminal_pass"

    def _load(self, context, model_dict, model_path):
        if model_path is None or model_path == "":
            return
        logging.info("load persistables from {}".format(model_path))

        with paddle.static.scope_guard(context["model"][model_dict["name"]][
                "scope"]):
            train_prog = context["model"][model_dict["name"]]["main_program"]
            startup_prog = context["model"][model_dict["name"]][
                "startup_program"]
            with paddle.static.program_guard(train_prog, startup_prog):
                fluid.io.load_persistables(
                    context["exe"], model_path, main_program=train_prog)
                self.auc_clear(context)

    def auc_clear(self, context):
        auc_states_names = [
            '_generated_var_0', '_generated_var_1', '_generated_var_2',
            '_generated_var_3'
        ]
        for name in auc_states_names:
            param = fluid.global_scope().var(name).get_tensor()
            if param:
                param_array = np.zeros(param._get_dims()).astype("int64")
                param.set(param_array, context['place'])
