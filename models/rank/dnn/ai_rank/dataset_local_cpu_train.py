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
import logging
import paddle
import paddle.fluid as fluid
from paddlerec.core.utils import envs
from paddlerec.core.trainers.framework.instance import InstanceBase
from paddlerec.core.trainers.framework.dataset import DataLoader, QueueDataset

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Instance(InstanceBase):
    "Ai-Rank train.py for dataset-local-cpu mode"

    def __init__(self, context):
        self.model = None
        self.input_data = None
        context["model"] = {}

    def instance(self, context):
        "Runtime main function"
        self.network(context)
        self.get_reader(context)
        self.run_startup(context)
        self.run_main(context)
        context['is_exit'] = True

    def network(self, context):
        " In AI-Rank, we assume only one phases will be implemented"
        self.model_dict = context["env"]["phase"][0]

        # 1. Get Model implement
        context["model"][self.model_dict["name"]] = {}
        model_path = envs.os_path_adapter(
            envs.workspace_adapter(self.model_dict["model"]))
        self.model = envs.lazy_instance_by_fliename(
            model_path, "Model")(context["env"])

        # 2. Set model input variable
        self.model._data_var = self.model.input_data(
            dataset_name=self.model_dict["dataset_name"])
        self.model.net(self.model._data_var, context["is_infer"])
        context["model"][self.model_dict["name"]]["model"] = self.model

        # 3. Append Backward & Optimize
        optimizer = self.model.optimizer()
        optimizer.minimize(self.model._cost)
        logger.info("Network pass end, build network success.")

    def get_reader(self, context):
        " In AI-Rank, we assume only one phases will be implemented"
        dataset_class = QueueDataset(context)
        self.reader = dataset_class.create_dataset(
            self.model_dict["dataset_name"], context)

    def run_startup(self, context):
        context["exe"].run(paddle.static.default_startup_program())

    def run_main(self, context):
        fetch_vars = self.model._metrics.values()
        fetch_alias = self.model._metrics.keys()
        fetch_period = int(
            envs.get_global_env("runner." + context["runner_name"] +
                                ".print_interval", 20))

        epochs = int(
            envs.get_global_env("runner." + context["runner_name"] +
                                ".epochs"))

        for epoch in range(epochs):
            context["exe"].train_from_dataset(
                program=paddle.static.default_main_program(),
                fetch_list=fetch_vars,
                fetch_info=fetch_alias,
                dataset=self.reader,
                print_period=fetch_period,
                debug=envs.get_global_env("debug", False))
