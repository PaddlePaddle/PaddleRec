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

from paddlerec.core.utils import envs

trainer_abs = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "trainers")
trainers = {}


def trainer_registry():
    trainers["SingleTrainer"] = os.path.join(trainer_abs, "single_trainer.py")
    trainers["ClusterTrainer"] = os.path.join(trainer_abs,
                                              "cluster_trainer.py")
    trainers["CtrCodingTrainer"] = os.path.join(trainer_abs,
                                                "ctr_coding_trainer.py")
    trainers["CtrModulTrainer"] = os.path.join(trainer_abs,
                                               "ctr_modul_trainer.py")
    trainers["TDMSingleTrainer"] = os.path.join(trainer_abs,
                                                "tdm_single_trainer.py")
    trainers["TDMClusterTrainer"] = os.path.join(trainer_abs,
                                                 "tdm_cluster_trainer.py")
    trainers["SingleTrainerYamlOpt"] = os.path.join(trainer_abs,
                                                 "single_trainer_yamlopt.py")
    trainers["SingleAucYamlOpt"] = os.path.join(trainer_abs,
                                                 "single_auc_yamlopt.py")

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
        _config = None
        if os.path.isfile(config):
            with open(config, 'r') as rb:
                _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
        else:
            raise ValueError("paddlerec's config only support yaml")

        envs.set_global_envs(_config)
        envs.update_workspace()

        trainer = TrainerFactory._build_trainer(config)
        return trainer


# server num, worker num
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("need a yaml file path argv")
    trainer = TrainerFactory.create(sys.argv[1])
    trainer.run()
