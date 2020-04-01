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
# limitations under the License.# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import yaml

from .single_train import SingleTrainerWithDataloader
from .single_train import SingleTrainerWithDataset

from .cluster_train import ClusterTrainerWithDataloader
from .cluster_train import ClusterTrainerWithDataset

from .ctr_trainer import CtrPaddleTrainer

from ..utils import envs


class TrainerFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def _build_trainer(config):
        train_mode = envs.get_global_env("train.trainer")
        reader_mode = envs.get_global_env("train.reader.mode")
        if train_mode == "SingleTraining":
            if reader_mode == "dataset":
                trainer = SingleTrainerWithDataset()
            elif reader_mode == "dataloader":
                trainer = SingleTrainerWithDataloader()
            else:
                raise ValueError("reader only support dataset/dataloader")
        elif train_mode == "ClusterTraining":
            if reader_mode == "dataset":
                trainer = ClusterTrainerWithDataset()
            elif reader_mode == "dataloader":
                trainer = ClusterTrainerWithDataloader()
            else:
                raise ValueError("reader only support dataset/dataloader")
        elif train_mode == "CtrTrainer":
            trainer = CtrPaddleTrainer(config)
        else:
            raise ValueError("trainer only support SingleTraining/ClusterTraining")

        return trainer

    @staticmethod
    def create(config):
        _config = None
        if isinstance(config, dict):
            _config = config
        elif isinstance(config, str):
            if os.path.exists(config) and os.path.isfile(config):
                with open(config, 'r') as rb:
                    _config = yaml.load(rb.read())
        else:
            raise ValueError("unknown config about eleps")

        envs.set_global_envs(_config)
        trainer = TrainerFactory._build_trainer(_config)

        return trainer
