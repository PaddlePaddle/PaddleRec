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
import warnings
import logging
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet
import config

from w2v_network import Model
from w2v_reader import Generator

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Main(object):
    def __init__(self):
        self.metrics = {}
        self.input_var = None
        self.reader = None
        self.exe = None

    def run(self):
        fleet.init()
        self.network()
        if fleet.is_server():
            self.run_server()
        elif fleet.is_worker():
            self.run_worker()
            fleet.stop_worker()
        logger.info("Run Success, Exit.")

    def network(self):
        model = Model()
        self.input_var = model.input_var()
        self.metrics = model.train_net(self.input_var)
        model.minimize(get_strategy())

    def run_server(self):
        logger.info("Run Server Begin")
        fleet.init_server(config.warmup_model_path)
        fleet.run_server()

    def run_worker(self):
        logger.info("Run Worker Begin")
        place = fluid.CUDAPlace(0) if config.use_cuda else fluid.CPUPlace()
        self.exe = fluid.Executor(place)

        self.exe.run(fluid.default_startup_program())
        fleet.init_worker()

        self.reader = get_reader(self.input_var, config.train_data_path)

        for epoch in range(config.epochs):
            if config.reader_type == "QueueDataset":
                self.dataset_train_loop(epoch)
            elif config.reader_type == "DataLoader":
                self.dataloader_train_loop(epoch)

    def dataset_train_loop(self, epoch):
        self.exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=self.reader,
            fetch_list=[var for _, var in self.metrics.items()],
            fetch_info=[
                "epoch {} Var {}".format(epoch, var_name)
                for var_name in self.metrics
            ],
            print_period=config.print_period,
            debug=config.dataset_debug)

    def dataloader_train_loop(self, epoch):
        logger.info("Epoch: {}, Running Begin.".format(epoch))
        for batch_id, data in enumerate(self.reader()):
            fetch_var = self.exe.run(
                program=fluid.default_main_program(),
                feed=data,
                fetch_list=[var for _, var in self.metrics.items()])
            if batch_id % config.print_period == 0:
                metrics_string = ""
                for var_idx, var_name in enumerate(self.metrics):
                    metrics_string += "{}: {}, ".format(var_name,
                                                        fetch_var[var_idx])
                logger.info("Epoch: {}, Batch: {}, {}".format(epoch, batch_id,
                                                              metrics_string))

    def record_profiler(self):
        pass


def get_strategy():
    if not is_distributed_env():
        logger.warn(
            "Not Find Distributed env, Change To local train mode. If you want train with fleet, please use [fleetrun] command."
        )
        return None
    assert config.sync_mode in ["async", "sync", "geo", "heter"]
    if config.sync_mode == "sync":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = False
    elif config.sync_mode == "async":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
    elif config.sync_mode == "geo":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"k_steps": config.geo_step}
    elif config.sync_mode == "heter":
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"heter_worker_device_guard": "gpu"}
    return strategy


def get_reader(input_var, data_path):
    assert config.reader_type in ["QueueDataset", "DataLoader"]
    file_list = get_file_list(data_path)
    if config.reader_type == "QueueDataset":
        reader_instance = QueueDatset(input_var, file_list)
        return reader_instance.get_reader()
    elif config.reader_type == "DataLoader":
        reader_instance = DataLoader(input_var, file_list)
        return reader_instance.get_reader()


def get_file_list(data_path):
    assert config.train_data_path != ""
    file_list = [
        config.train_data_path + "/%s" % x
        for x in os.listdir(config.train_data_path)
    ]
    if config.split_file_list:
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


def is_distributed_env():
    node_role = os.getenv("TRAINING_ROLE")
    logger.info("-- Role: {} --".format(node_role))
    if node_role is None:
        return False
    else:
        return True


class QueueDatset(object):
    def __init__(self, input_var, file_list):
        assert isinstance(input_var, list)
        assert len(file_list) > 0
        assert config.pipe_command != None
        self.input_var = input_var
        self.file_list = file_list

    def get_reader(self):
        logger.info("Get Dataset")
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var(self.input_var)
        dataset.set_pipe_command(config.pipe_command)
        dataset.set_batch_size(config.batch_size)
        dataset.set_thread(config.thread_num)
        dataset.set_filelist(self.file_list)
        return dataset


class DataLoader(object):
    def __init__(self, input_var, file_list):
        assert isinstance(input_var, list)
        assert len(file_list) > 0
        self.input_var = input_var
        self.file_list = file_list

    def get_reader(self):
        logger.info("Get DataLoader")
        loader = fluid.io.DataLoader.from_generator(
            feed_list=self.input_var, capacity=64, iterable=True)
        generator = Generator()
        generator.init()
        place = fluid.CUDAPlace(0) if config.use_cuda else fluid.CPUPlace()
        loader.set_sample_generator(
            generator.dataloader(self.file_list),
            batch_size=config.batch_size,
            drop_last=True,
            places=place)
        return loader


if __name__ == "__main__":
    paddle.enable_static()
    os.environ["CPU_NUM"] = str(config.thread_num)
    benchamrk_main = Main()
    benchamrk_main.run()
