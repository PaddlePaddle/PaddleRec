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
import warnings
from paddlerec.core.utils.envs import lazy_instance_by_fliename
from paddlerec.core.utils.envs import get_global_env
from paddlerec.core.utils.envs import get_runtime_environ
from paddlerec.core.reader import SlotReader
from paddlerec.core.trainer import EngineMode
from paddlerec.core.utils.util import split_files, check_filelist


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

    hidden_file_list, files = check_filelist(
        hidden_file_list=[], data_file_list=[], train_data_path=data_path)
    if (hidden_file_list is not None):
        print(
            "Warning:please make sure there are no hidden files in the dataset folder and check these hidden files:{}".
            format(hidden_file_list))

    files.sort()

    # for local cluster: discard some files if files cannot be divided equally between GPUs
    if (context["device"] == "GPU"
        ) and os.getenv("PADDLEREC_GPU_NUMS") is not None:
        selected_gpu_nums = int(os.getenv("PADDLEREC_GPU_NUMS"))
        discard_file_nums = len(files) % selected_gpu_nums
        if (discard_file_nums != 0):
            warnings.warn(
                "Because files cannot be divided equally between GPUs,discard these files:{}".
                format(files[-discard_file_nums:]))
            files = files[:len(files) - discard_file_nums]

    need_split_files = False
    if context["engine"] == EngineMode.LOCAL_CLUSTER:
        # for local cluster: split files for multi process
        need_split_files = True
    elif context["engine"] == EngineMode.CLUSTER and context[
            "cluster_type"] == "K8S":
        # for k8s mount mode, split files for every node
        need_split_files = True
    print("need_split_files: {}".format(need_split_files))
    if need_split_files:
        files = split_files(files, context["fleet"].worker_index(),
                            context["fleet"].worker_num())

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

    if hasattr(reader, "batch_tensor_creator"):
        return reader.batch_tensor_creator(gen_reader)

    return gen_reader


def slotdataloader_by_name(readerclass, dataset_name, yaml_file, context):
    name = "dataset." + dataset_name + "."
    reader_name = "SlotReader"
    data_path = get_global_env(name + "data_path")

    if data_path.startswith("paddlerec::"):
        package_base = get_runtime_environ("PACKAGE_BASE")
        assert package_base is not None
        data_path = os.path.join(package_base, data_path.split("::")[1])

    hidden_file_list, files = check_filelist(
        hidden_file_list=[], data_file_list=[], train_data_path=data_path)
    if (hidden_file_list is not None):
        print(
            "Warning:please make sure there are no hidden files in the dataset folder and check these hidden files:{}".
            format(hidden_file_list))

    files.sort()

    # for local cluster: discard some files if files cannot be divided equally between GPUs
    if (context["device"] == "GPU"
        ) and os.getenv("PADDLEREC_GPU_NUMS") is not None:
        selected_gpu_nums = int(os.getenv("PADDLEREC_GPU_NUMS"))
        discard_file_nums = len(files) % selected_gpu_nums
        if (discard_file_nums != 0):
            warnings.warn(
                "Because files cannot be divided equally between GPUs,discard these files:{}".
                format(files[-discard_file_nums:]))
            files = files[:len(files) - discard_file_nums]

    need_split_files = False
    if context["engine"] == EngineMode.LOCAL_CLUSTER:
        # for local cluster: split files for multi process
        need_split_files = True
    elif context["engine"] == EngineMode.CLUSTER and context[
            "cluster_type"] == "K8S":
        # for k8s mount mode, split files for every node
        need_split_files = True

    if need_split_files:
        files = split_files(files, context["fleet"].worker_index(),
                            context["fleet"].worker_num())

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

    hidden_file_list, files = check_filelist(
        hidden_file_list=[], data_file_list=[], train_data_path=data_path)
    if (hidden_file_list is not None):
        print(
            "Warning:please make sure there are no hidden files in the dataset folder and check these hidden files:{}".
            format(hidden_file_list))

    files.sort()

    # for local cluster: discard some files if files cannot be divided equally between GPUs
    if (context["device"] == "GPU"
        ) and os.getenv("PADDLEREC_GPU_NUMS") is not None:
        selected_gpu_nums = int(os.getenv("PADDLEREC_GPU_NUMS"))
        discard_file_nums = len(files) % selected_gpu_nums
        if (discard_file_nums != 0):
            warnings.warn(
                "Because files cannot be divided equally between GPUs,discard these files:{}".
                format(files[-discard_file_nums:]))
            files = files[:len(files) - discard_file_nums]

    need_split_files = False
    if context["engine"] == EngineMode.LOCAL_CLUSTER:
        # for local cluster: split files for multi process
        need_split_files = True
    elif context["engine"] == EngineMode.CLUSTER and context[
            "cluster_type"] == "K8S":
        # for k8s mount mode, split files for every node
        need_split_files = True

    if need_split_files:
        files = split_files(files, context["fleet"].worker_index(),
                            context["fleet"].worker_num())

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
