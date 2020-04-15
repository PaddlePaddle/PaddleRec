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

import os
import sys

import yaml

from fleetrec.trainer.local_engine import Launch
from fleetrec.trainer.single_trainer import SingleTrainer
from fleetrec.trainer.cluster_trainer import ClusterTrainer
from fleetrec.trainer.ctr_trainer import CtrPaddleTrainer

from fleetrec.utils import envs


class TrainerFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def _build_trainer(config, yaml_path):
        print(envs.pretty_print_envs(envs.get_global_envs()))

        train_mode = envs.get_runtime_envion("train.trainer")

        if train_mode == "SingleTraining":
            trainer = SingleTrainer(yaml_path)
        elif train_mode == "ClusterTraining":
            trainer = ClusterTrainer(yaml_path)
        elif train_mode == "CtrTraining":
            trainer = CtrPaddleTrainer(config)
        elif train_mode == "UserDefineTraining":
            train_location = envs.get_global_env("train.location")
            train_dirname = os.path.dirname(train_location)
            base_name = os.path.splitext(os.path.basename(train_location))[0]
            sys.path.append(train_dirname)
            trainer_class = envs.lazy_instance(base_name, "UserDefineTrainer")
            trainer = trainer_class(yaml_path)
        else:
            raise ValueError("trainer only support SingleTraining/ClusterTraining")
        return trainer

    @staticmethod
    def create(config):
        _config = None
        if os.path.exists(config) and os.path.isfile(config):
            with open(config, 'r') as rb:
                _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
        else:
            raise ValueError("fleetrec's config only support yaml")

        envs.set_global_envs(_config)
        trainer = TrainerFactory._build_trainer(_config, config)
        return trainer


# server num, worker num
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("need a yaml file path argv")
    trainer = TrainerFactory.create(sys.argv[1])
    trainer.run()
