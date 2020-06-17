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

import paddle.fluid as fluid
from paddlerec.core.utils import envs
from paddlerec.core.utils import dataloader_instance
from paddlerec.core.reader import SlotReader
from paddlerec.core.trainer import EngineMode
from paddlerec.core.utils.util import split_files

__all__ = ["DatasetBase", "DataLoader", "QueueDataset"]


class DatasetBase(object):
    """R
    """

    def __init__(self, context):
        pass

    def get_dataset(self, context):
        pass


class DataLoader(DatasetBase):
    def __init__(self, context):
        pass

    def get_dataloader(self, context, dataset_name, dataloader):
        name = "dataset." + dataset_name + "."
        sparse_slots = envs.get_global_env(name + "sparse_slots", "").strip()
        dense_slots = envs.get_global_env(name + "dense_slots", "").strip()
        batch_size = envs.get_global_env(name + "batch_size")

        reader_class = envs.get_global_env(name + "data_converter")
        reader_class_name = envs.get_global_env(name + "reader_class_name",
                                                "Reader")

        if sparse_slots == "" and dense_slots == "":
            reader = dataloader_instance.dataloader_by_name(
                reader_class,
                dataset_name,
                context["config_yaml"],
                context,
                reader_class_name=reader_class_name)

            reader_class = envs.lazy_instance_by_fliename(reader_class,
                                                          reader_class_name)
            reader_ins = reader_class(context["config_yaml"])
        else:
            reader = dataloader_instance.slotdataloader_by_name(
                "", dataset_name, context["config_yaml"], context)
            reader_ins = SlotReader(context["config_yaml"])
        if hasattr(reader_ins, 'generate_batch_from_trainfiles'):
            dataloader.set_sample_list_generator(reader)
        else:
            dataloader.set_sample_generator(reader, batch_size)
        return dataloader


class QueueDataset(DatasetBase):
    def __init__(self, context):
        pass

    def create_dataset(self, dataset_name, context):
        name = "dataset." + dataset_name + "."
        type_name = envs.get_global_env(name + "type")
        if envs.get_platform() != "LINUX":
            print("platform ", envs.get_platform(), "Reader To Dataloader")
            type_name = "DataLoader"

        if type_name == "DataLoader":
            return None
        else:
            return self._get_dataset(dataset_name, context)

    def _get_dataset(self, dataset_name, context):
        name = "dataset." + dataset_name + "."
        reader_class = envs.get_global_env(name + "data_converter")
        reader_class_name = envs.get_global_env(name + "reader_class_name",
                                                "Reader")
        abs_dir = os.path.dirname(os.path.abspath(__file__))
        reader = os.path.join(abs_dir, '../../utils', 'dataset_instance.py')
        sparse_slots = envs.get_global_env(name + "sparse_slots", "").strip()
        dense_slots = envs.get_global_env(name + "dense_slots", "").strip()
        if sparse_slots == "" and dense_slots == "":
            pipe_cmd = "python {} {} {} {}".format(reader, reader_class,
                                                   reader_class_name,
                                                   context["config_yaml"])
        else:
            if sparse_slots == "":
                sparse_slots = "?"
            if dense_slots == "":
                dense_slots = "?"
            padding = envs.get_global_env(name + "padding", 0)
            pipe_cmd = "python {} {} {} {} {} {} {} {}".format(
                reader, "slot", "slot", context["config_yaml"], "fake",
                sparse_slots.replace(" ", "?"),
                dense_slots.replace(" ", "?"), str(padding))

        batch_size = envs.get_global_env(name + "batch_size")
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_batch_size(batch_size)
        dataset.set_pipe_command(pipe_cmd)
        train_data_path = envs.get_global_env(name + "data_path")
        file_list = [
            os.path.join(train_data_path, x)
            for x in os.listdir(train_data_path)
        ]
        if context["engine"] == EngineMode.LOCAL_CLUSTER:
            file_list = split_files(file_list, context["fleet"].worker_index(),
                                    context["fleet"].worker_num())

        dataset.set_filelist(file_list)
        for model_dict in context["phases"]:
            if model_dict["dataset_name"] == dataset_name:
                model = context["model"][model_dict["name"]]["model"]
                thread_num = int(model_dict["thread_num"])
                dataset.set_thread(thread_num)
                if context["is_infer"]:
                    inputs = model._infer_data_var
                else:
                    inputs = model._data_var
                dataset.set_use_var(inputs)
                break
        return dataset
