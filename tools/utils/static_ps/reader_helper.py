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
import sys
import warnings
import logging
import paddle
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
from . import common_ps

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_reader(input_var, config):
    reader_type = config.get("runner.reader_type")
    train_data_path = config.get("runner.train_data_dir")
    assert train_data_path != ""

    train_data_path = os.path.join(config["config_abs_dir"], train_data_path)

    assert reader_type in [
        "QueueDataset", "DataLoader", "RecDataset", "InmemoryDataset", None
    ]
    file_list = get_file_list(train_data_path, config)
    print("train file_list: {}".format(file_list))
    if reader_type == "QueueDataset":
        reader_instance = QueueDatset(input_var, file_list, config)
        return reader_instance.get_reader(), file_list
    elif reader_type == "InmemoryDataset":
        reader_instance = InmemoryDatset(input_var, file_list, config)
        return reader_instance.get_reader(), file_list
    elif reader_type == "DataLoader":
        reader_instance = DataLoader(input_var, file_list, config)
        return reader_instance.get_reader(), file_list
    elif reader_type == None or reader_type == "RecDataset":
        reader_instance = RecDatasetReader(input_var, file_list, config)
        return reader_instance.get_reader(), file_list


def get_infer_reader(input_var, config):
    test_data_path = config.get("runner.test_data_dir")
    assert test_data_path != ""
    test_data_path = os.path.join(config["config_abs_dir"], test_data_path)
    print("test_data_path is: {}".format(test_data_path))
    file_list = get_file_list(test_data_path, config)
    print("test file_list: {}".format(file_list))
    reader_type = config.get("runner.infer_reader_type")
    if reader_type == "QueueDataset":
        reader_instance = QueueDatset(input_var, file_list, config)
        return reader_instance.get_infer_reader(), file_list
    else:
        reader_instance = InferDataLoader(input_var, file_list, config)
        return reader_instance.get_reader(), file_list


def get_file_list(data_path, config):
    assert os.path.exists(data_path)
    file_list = [data_path + "/%s" % x for x in os.listdir(data_path)]
    if config.get("runner.split_file_list"):
        logger.info("Split file list for worker {}".format(fleet.worker_index(
        )))
        file_list = fleet.util.get_file_shard(file_list)
    logger.info("File list: {}".format(file_list))
    return file_list


def get_example_num(file_list):
    count = 0
    for f in file_list:
        last_count = count
        for _, _ in enumerate(open(f, 'r')):
            count += 1
        logger.info("File: %s has %s examples" % (f, count - last_count))
    logger.info("Total example: %s" % count)
    return count


def get_word_num(file_list):
    count = 0
    for f in file_list:
        last_count = count
        for index, line in enumerate(open(f, 'r')):
            line = line.rstrip().split()
            count += len(line)
        logger.info("file: %s has %s words" % (f, count - last_count))
    logger.info("Total words: %s" % count)
    return count


def get_reader_generator(path, reader_name="Reader"):
    reader_class = common_ps.lazy_instance_by_fliename(path, reader_name)()
    return reader_class


class RecDatasetReader(object):
    def __init__(self, input_var, file_list, config):
        assert isinstance(input_var, list)
        assert len(file_list) > 0
        self.input_var = input_var
        self.file_list = file_list
        self.config = config

    def get_reader(self):
        logger.info("Get DataLoader")

        config_abs_dir = self.config.get("config_abs_dir", None)
        reader_path = self.config.get('runner.train_reader_path', 'reader')
        reader_path = os.path.join(config_abs_dir, reader_path)
        logger.info("Reader Path: {}".format(reader_path))

        from paddle.io import DataLoader
        dataset = common_ps.lazy_instance_by_fliename(reader_path,
                                                      "RecDataset")
        print("dataset: {}".format(dataset))

        use_cuda = int(self.config.get("runner.use_gpu"))
        batch_size = self.config.get('runner.train_batch_size', None)
        place = paddle.set_device('gpu' if use_cuda else 'cpu')

        generator = dataset(self.file_list, self.config)
        generator.init()
        loader = DataLoader(
            generator, batch_size=batch_size, places=place, drop_last=True)
        return loader


class DataLoader(object):
    def __init__(self, input_var, file_list, config):
        assert isinstance(input_var, list)
        assert len(file_list) > 0
        self.input_var = input_var
        self.file_list = file_list
        self.config = config

    def get_reader(self):
        logger.info("Get DataLoader")
        loader = paddle.io.DataLoader.from_generator(
            feed_list=self.input_var,
            capacity=64,
            iterable=False,
            use_double_buffer=False)
        path = self.config.get("runner.train_reader_path")
        path = os.path.join(self.config["config_abs_dir"], path)
        generator = get_reader_generator(path)
        generator.init(self.config)
        batch_size = int(self.config.get("runner.train_batch_size"))
        batch_generator = self.config.get("runner.batch_generator", False)
        if batch_generator:
            loader.set_batch_generator(generator.dataloader(self.file_list))
        else:
            loader.set_sample_generator(
                generator.dataloader(self.file_list),
                batch_size=batch_size,
                drop_last=True,
                places=paddle.static.cpu_places())
        return loader


class InferDataLoader(object):
    def __init__(self, input_var, file_list, config):
        assert isinstance(input_var, list)
        assert len(file_list) > 0
        self.input_var = input_var
        self.file_list = file_list
        self.config = config

    def get_reader(self):
        logger.info("Get DataLoader")
        use_cuda = int(self.config.get("runner.use_gpu"))
        place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
        data_dir = self.config.get("runner.test_data_dir", None)
        batch_size = self.config.get('runner.infer_batch_size', None)
        reader_path = self.config.get('runner.infer_reader_path', 'reader')
        num_workers = int(self.config.get('runner.num_workers', 0))
        config_abs_dir = self.config.get("config_abs_dir", None)
        data_dir = os.path.join(config_abs_dir, data_dir)
        file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
        user_define_reader = self.config.get('runner.user_define_reader',
                                             False)
        logger.info("reader path:{}".format(reader_path))
        from importlib import import_module
        reader_class = import_module(reader_path)
        dataset = reader_class.RecDataset(file_list, config=self.config)
        loader = paddle.io.DataLoader(
            dataset,
            batch_size=batch_size,
            places=place,
            drop_last=True,
            num_workers=num_workers)
        return loader


class QueueDatset(object):
    def __init__(self, input_var, file_list, config):
        assert isinstance(input_var, list)
        assert len(file_list) > 0
        self.config = config
        self.input_var = input_var
        self.file_list = file_list
        self.parse_ins_id = self.config.get("runner.parse_ins_id")
        print("parse ins id:", self.parse_ins_id)
        self.pipe_command = self.config.get("runner.pipe_command")
        self.train_reader = self.config.get("runner.train_reader_path")
        assert self.pipe_command != None
        utils_path = common_ps.get_utils_file_path()
        print("utils_path: {}".format(utils_path))
        abs_train_reader = os.path.join(config["config_abs_dir"],
                                        self.train_reader)
        print("abs_train_reader is: {}".format(abs_train_reader))
        self.pipe_command = self.pipe_command.replace(self.train_reader,
                                                      abs_train_reader)
        self.pipe_command = "{} {} {}".format(self.pipe_command,
                                              config.get("yaml_path"),
                                              utils_path)
        print("pipe_command is: {}".format(self.pipe_command))
        self.batch_size = int(self.config.get("runner.train_batch_size"))
        assert self.batch_size >= 1
        self.thread_num = int(self.config.get("runner.thread_num"))
        print("dataset init thread_num:", self.thread_num)
        assert self.thread_num >= 1

    def get_reader(self):
        logger.info("Get Train Dataset")
        dataset = paddle.distributed.QueueDataset()
        dataset.init(
            use_var=self.input_var,
            pipe_command=self.pipe_command,
            batch_size=self.batch_size,
            thread_num=self.thread_num)
        print("dataset get_reader thread_num:", self.thread_num)
        dataset.set_filelist(self.file_list)
        return dataset

    def get_infer_reader(self):
        logger.info("Get Infer Dataset")
        dataset = paddle.distributed.QueueDataset()
        self.infer_batch_size = int(self.config.get("runner.infer_batch_size"))
        self.infer_thread_num = self.thread_num
        dataset.init(
            use_var=self.input_var,
            pipe_command=self.pipe_command,
            batch_size=self.infer_batch_size,
            thread_num=self.infer_thread_num)
        print("dataset get_infer_reader thread_num:", self.infer_thread_num)
        dataset.set_filelist(self.file_list)
        return dataset


class InmemoryDatset(object):
    def __init__(self, input_var, file_list, config):
        assert isinstance(input_var, list)
        assert len(file_list) > 0
        self.config = config
        self.input_var = input_var
        self.file_list = file_list
        self.pipe_command = self.config.get("runner.pipe_command")
        self.train_reader = self.config.get("runner.train_reader_path")
        assert self.pipe_command != None
        utils_path = common_ps.get_utils_file_path()
        print("utils_path: {}".format(utils_path))
        abs_train_reader = os.path.join(config["config_abs_dir"],
                                        self.train_reader)
        self.pipe_command = self.pipe_command.replace(self.train_reader,
                                                      abs_train_reader)
        self.pipe_command = "{} {} {}".format(self.pipe_command,
                                              config.get("yaml_path"),
                                              utils_path)
        print(self.pipe_command)
        self.batch_size = int(self.config.get("runner.train_batch_size"))
        assert self.batch_size >= 1
        self.thread_num = int(self.config.get("runner.thread_num"))
        assert self.thread_num >= 1
        self.parse_ins_id = self.config.get("runner.parse_ins_id", False)
        self.parse_content = self.config.get("runner.parse_content", False)
        self.fs_name = self.config.get("runner.fs_name", "")
        self.fs_ugi = self.config.get("runner.fs_ugi", "")
        print("hdfs config:", self.fs_name, self.fs_ugi)
        self.use_gpu = self.config.get("runner.use_gpu", False)

    def get_reader(self):
        logger.info("Get InmemoryDataset")
        dataset = paddle.distributed.InMemoryDataset()
        dataset._set_use_ps_gpu(self.use_gpu)
        dataset.init(
            use_var=self.input_var,
            pipe_command=self.pipe_command,
            batch_size=self.batch_size,
            thread_num=self.thread_num,
            fs_name=self.fs_name,
            fs_ugi=self.fs_ugi)
        dataset.set_filelist(self.file_list)
        dataset.update_settings(
            parse_ins_id=self.parse_ins_id, parse_content=self.parse_content)
        return dataset
