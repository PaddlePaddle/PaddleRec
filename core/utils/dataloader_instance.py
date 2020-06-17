#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddlerec.core.utils.envs import lazy_instance_by_fliename
from paddlerec.core.utils.envs import get_global_env
from paddlerec.core.utils.envs import get_runtime_environ
from paddlerec.core.reader import SlotReader
from paddlerec.core.trainer import EngineMode
from paddlerec.core.utils.util import split_files


def dataloader_by_name(readerclass,
                       dataset_name,
                       yaml_file,
                       context,
                       reader_class_name="Reader"):

    reader_class = lazy_instance_by_fliename(readerclass, reader_class_name)

    name = "dataset." + dataset_name + "."
    data_path = get_global_env(name + "data_path")

    if data_path.startswith("paddlerec::"):
        package_base = get_runtime_environ("PACKAGE_BASE")
        assert package_base is not None
        data_path = os.path.join(package_base, data_path.split("::")[1])

    files = [str(data_path) + "/%s" % x for x in os.listdir(data_path)]
    if context["engine"] == EngineMode.LOCAL_CLUSTER:
        files = split_files(files, context["fleet"].worker_index(),
                            context["fleet"].worker_num())
        print("file_list : {}".format(files))

    reader = reader_class(yaml_file)
    reader.init()

    def gen_reader():
        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    line = line.rstrip('\n')
                    iter = reader.generate_sample(line)
                    for parsed_line in iter():
                        if parsed_line is None:
                            continue
                        else:
                            values = []
                            for pased in parsed_line:
                                values.append(pased[1])
                            yield values

    def gen_batch_reader():
        return reader.generate_batch_from_trainfiles(files)

    if hasattr(reader, 'generate_batch_from_trainfiles'):
        return gen_batch_reader()
    return gen_reader


def slotdataloader_by_name(readerclass, dataset_name, yaml_file, context):
    name = "dataset." + dataset_name + "."
    reader_name = "SlotReader"
    data_path = get_global_env(name + "data_path")

    if data_path.startswith("paddlerec::"):
        package_base = get_runtime_environ("PACKAGE_BASE")
        assert package_base is not None
        data_path = os.path.join(package_base, data_path.split("::")[1])

    files = [str(data_path) + "/%s" % x for x in os.listdir(data_path)]
    if context["engine"] == EngineMode.LOCAL_CLUSTER:
        files = split_files(files, context["fleet"].worker_index(),
                            context["fleet"].worker_num())
        print("file_list: {}".format(files))

    sparse = get_global_env(name + "sparse_slots", "#")
    if sparse == "":
        sparse = "#"
    dense = get_global_env(name + "dense_slots", "#")
    if dense == "":
        dense = "#"
    padding = get_global_env(name + "padding", 0)
    reader = SlotReader(yaml_file)
    reader.init(sparse, dense, int(padding))

    def gen_reader():
        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    line = line.rstrip('\n')
                    iter = reader.generate_sample(line)
                    for parsed_line in iter():
                        if parsed_line is None:
                            continue
                        else:
                            values = []
                            for pased in parsed_line:
                                values.append(pased[1])
                            yield values

    def gen_batch_reader():
        return reader.generate_batch_from_trainfiles(files)

    if hasattr(reader, 'generate_batch_from_trainfiles'):
        return gen_batch_reader()
    return gen_reader


def slotdataloader(readerclass, train, yaml_file, context):
    if train == "TRAIN":
        reader_name = "SlotReader"
        namespace = "train.reader"
        data_path = get_global_env("train_data_path", None, namespace)
    else:
        reader_name = "SlotReader"
        namespace = "evaluate.reader"
        data_path = get_global_env("test_data_path", None, namespace)

    if data_path.startswith("paddlerec::"):
        package_base = get_runtime_environ("PACKAGE_BASE")
        assert package_base is not None
        data_path = os.path.join(package_base, data_path.split("::")[1])

    files = [str(data_path) + "/%s" % x for x in os.listdir(data_path)]
    if context["engine"] == EngineMode.LOCAL_CLUSTER:
        files = split_files(files, context["fleet"].worker_index(),
                            context["fleet"].worker_num())
        print("file_list: {}".format(files))

    sparse = get_global_env("sparse_slots", "#", namespace)
    if sparse == "":
        sparse = "#"
    dense = get_global_env("dense_slots", "#", namespace)
    if dense == "":
        dense = "#"
    padding = get_global_env("padding", 0, namespace)
    reader = SlotReader(yaml_file)
    reader.init(sparse, dense, int(padding))

    def gen_reader():
        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    line = line.rstrip('\n')
                    iter = reader.generate_sample(line)
                    for parsed_line in iter():
                        if parsed_line is None:
                            continue
                        else:
                            values = []
                            for pased in parsed_line:
                                values.append(pased[1])
                            yield values

    def gen_batch_reader():
        return reader.generate_batch_from_trainfiles(files)

    if hasattr(reader, 'generate_batch_from_trainfiles'):
        return gen_batch_reader()
    return gen_reader
