#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from .trainer_desc import MultiTrainer, DistMultiTrainer, PipelineTrainer
from .device_worker import Hogwild, DownpourSGD, Section

__all__ = ["TrainerFactory"]


class TrainerFactory(object):
    def __init__(self):
        pass

    def _create_trainer(self, opt_info=None):
        trainer = None
        device_worker = None
        if opt_info == None:
            # default is MultiTrainer + Hogwild
            trainer = MultiTrainer()
            device_worker = Hogwild()
            trainer._set_device_worker(device_worker)
        else:
            trainer_class = opt_info["trainer"]
            device_worker_class = opt_info["device_worker"]
            trainer = globals()[trainer_class]()
            device_worker = globals()[device_worker_class]()
            if "fleet_desc" in opt_info:
                device_worker._set_fleet_desc(opt_info["fleet_desc"])
                trainer._set_fleet_desc(opt_info["fleet_desc"])
                if opt_info.get("use_cvm") is not None:
                    trainer._set_use_cvm(opt_info["use_cvm"])
                if opt_info.get("scale_datanorm") is not None:
                    trainer._set_scale_datanorm(opt_info["scale_datanorm"])
                if opt_info.get("dump_slot") is not None:
                    trainer._set_dump_slot(opt_info["dump_slot"])
                if opt_info.get("mpi_rank") is not None:
                    trainer._set_mpi_rank(opt_info["mpi_rank"])
                if opt_info.get("dump_fields") is not None:
                    trainer._set_dump_fields(opt_info["dump_fields"])
                if opt_info.get("dump_fields_path") is not None:
                    trainer._set_dump_fields_path(opt_info["dump_fields_path"])
                if opt_info.get("user_define_dump_filename") is not None:
                    trainer._set_user_define_dump_filename(opt_info["user_define_dump_filename"])
                if opt_info.get("dump_converter") is not None:
                    trainer._set_dump_converter(opt_info["dump_converter"])
                if opt_info.get("adjust_ins_weight") is not None:
                    trainer._set_adjust_ins_weight(opt_info["adjust_ins_weight"])
            trainer._set_device_worker(device_worker)
        return trainer
