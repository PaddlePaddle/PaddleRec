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
from paddlerec.core.utils import envs

trainer_abs = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "trainers")
trainers = {}


def trainer_registry():
    # Definition of procedure execution process
    trainers["CtrCodingTrainer"] = os.path.join(trainer_abs,
                                                "ctr_coding_trainer.py")
    trainers["CtrModulTrainer"] = os.path.join(trainer_abs,
                                               "ctr_modul_trainer.py")
    trainers["GeneralTrainer"] = os.path.join(trainer_abs,
                                              "general_trainer.py")


trainer_registry()


class TrainerFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def _build_trainer(yaml_path):
        print(envs.pretty_print_envs(envs.get_global_envs()))

        train_mode = envs.get_trainer()
        trainer_abs = trainers.get(train_mode, None)

        if trainer_abs is None:
            if not os.path.isfile(train_mode):
                raise IOError("trainer {} can not be recognized".format(
                    train_mode))
            trainer_abs = train_mode
            train_mode = "UserDefineTrainer"

        trainer_class = envs.lazy_instance_by_fliename(trainer_abs, train_mode)
        trainer = trainer_class(yaml_path)
        return trainer

    @staticmethod
    def create(config):
        _config = envs.load_yaml(config)
        envs.set_global_envs(_config, True)
        trainer = TrainerFactory._build_trainer(config)
        return trainer


# server num, worker num
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("need a yaml file path argv")
    trainer = TrainerFactory.create(sys.argv[1])
    trainer.run()
