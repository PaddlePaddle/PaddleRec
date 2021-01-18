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
import utils
import time
import reader
import program
import argparse

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("PaddleRec train script")
    parser.add_argument(
        '-c',
        '--config_yaml',
        type=str,
        required=True,
        help='config file path')
    args = parser.parse_args()
    yaml_helper = utils.YamlHelper()
    config = yaml_helper.load_yaml(args.config_yaml)
    config["yaml_path"] = args.config_yaml
    yaml_helper.print_yaml(config)
    return config


class Main(object):
    def __init__(self, config):
        self.metrics = {}
        self.config = config
        self.input_data = None
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
        model = program.get_model(self.config)
        self.input_data = model.input_data()
        self.init_reader()
        self.metrics = model.net(self.input_data)
        self.infer_target_var = model.infer_target_var
        logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))
        model.minimize(program.get_strategy(self.config))

    def run_server(self):
        logger.info("Run Server Begin")
        fleet.init_server(config.get("static_benchmark.warmup_model_path"))
        fleet.run_server()

    def run_worker(self):
        logger.info("Run Worker Begin")
        use_cuda = int(config.get("static_benchmark.use_cuda"))
        place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
        self.exe = paddle.static.Executor(place)

        with open("./{}_worker_main_program.prototxt".format(
                fleet.worker_index()), 'w+') as f:
            f.write(str(paddle.static.default_main_program()))
        with open("./{}_worker_startup_program.prototxt".format(
                fleet.worker_index()), 'w+') as f:
            f.write(str(paddle.static.default_startup_program()))

        self.exe.run(paddle.static.default_startup_program())
        fleet.init_worker()

        save_model_path = self.config.get("static_benchmark.save_model_path")
        if save_model_path and not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        reader_type = self.config.get("static_benchmark.reader_type")
        epochs = int(self.config.get("static_benchmark.epochs"))
        sync_mode = self.config.get("static_benchmark.sync_mode")

        for epoch in range(epochs):
            epoch_start_time = time.time()

            if sync_mode == "heter":
                self.heter_train_loop(epoch)
            elif reader_type == "QueueDataset":
                self.dataset_train_loop(epoch)
            elif reader_type == "DataLoader":
                self.dataloader_train_loop(epoch)

            epoch_time = time.time() - epoch_start_time
            epoch_speed = self.example_nums / epoch_time
            logger.info(
                "Epoch: {}, using time {} second, ips {} {}/sec.".format(
                    epoch, epoch_time, epoch_speed, self.count_method))

            model_dir = "{}/{}".format(save_model_path, epoch)
            if fleet.is_first_worker() and save_model_path:
                fleet.save_inference_model(
                    self.exe, model_dir,
                    [feed.name for feed in self.input_data],
                    self.infer_target_var)

    def init_reader(self):
        if fleet.is_server():
            return
        self.reader, self.file_list = reader.get_reader(self.input_data,
                                                        config)
        self.example_nums = 0
        self.count_method = self.config.get(
            "static_benchmark.example_count_method", "example")
        if self.count_method == "example":
            self.example_nums = reader.get_example_num(self.file_list)
        elif self.count_method == "word":
            self.example_nums = reader.get_word_num(self.file_list)
        else:
            raise ValueError(
                "Set static_benchmark.example_count_method for example / word for example count."
            )

    def dataset_train_loop(self, epoch):
        logger.info("Epoch: {}, Running Begin.".format(epoch))
        fetch_info = [
            "Epoch {} Var {}".format(epoch, var_name)
            for var_name in self.metrics
        ]
        fetch_vars = [var for _, var in self.metrics.items()]
        print_step = int(config.get("static_benchmark.print_period"))
        self.exe.train_from_dataset(
            program=paddle.static.default_main_program(),
            dataset=self.reader,
            fetch_list=fetch_vars,
            fetch_info=fetch_info,
            print_period=print_step,
            debug=config.get("static_benchmark.dataset_debug"))

    def dataloader_train_loop(self, epoch):
        logger.info("Epoch: {}, Running Begin.".format(epoch))
        batch_id = 0
        train_run_cost = 0.0
        total_examples = 0
        self.reader.start()
        while True:
            try:
                train_start = time.time()
                # --------------------------------------------------- #
                fetch_var = self.exe.run(
                    program=paddle.static.default_main_program(),
                    fetch_list=[var for _, var in self.metrics.items()])
                # --------------------------------------------------- #
                train_run_cost += time.time() - train_start
                total_examples += self.config.get(
                    "static_benchmark.batch_size")
                batch_id += 1
                print_step = int(config.get("static_benchmark.print_period"))
                if batch_id % print_step == 0:
                    metrics_string = ""
                    for var_idx, var_name in enumerate(self.metrics):
                        metrics_string += "{}: {}, ".format(var_name,
                                                            fetch_var[var_idx])
                    profiler_string = ""
                    profiler_string += "avg_batch_cost: {} sec, ".format(
                        format((train_run_cost) / print_step, '.5f'))
                    profiler_string += "avg_samples: {}, ".format(
                        format(total_examples / print_step, '.5f'))
                    profiler_string += "ips: {} {}/sec ".format(
                        format(total_examples / (train_run_cost), '.5f'),
                        self.count_method)
                    logger.info("Epoch: {}, Batch: {}, {} {}".format(
                        epoch, batch_id, metrics_string, profiler_string))
                    train_run_cost = 0.0
                    total_examples = 0
            except fluid.core.EOFException:
                self.reader.reset()
                break

    def heter_train_loop(self, epoch):
        logger.info(
            "Epoch: {}, Running Begin. Check running metrics at heter_log".
            format(epoch))
        reader_type = self.config.get("static_benchmark.reader_type")
        if reader_type == "QueueDataset":
            self.exe.train_from_dataset(
                program=paddle.static.default_main_program(),
                dataset=self.reader,
                debug=config.get("static_benchmark.dataset_debug"))
        elif reader_type == "DataLoader":
            batch_id = 0
            train_run_cost = 0.0
            total_examples = 0
            self.reader.start()
            while True:
                try:
                    train_start = time.time()
                    # --------------------------------------------------- #
                    self.exe.run(program=paddle.static.default_main_program())
                    # --------------------------------------------------- #
                    train_run_cost += time.time() - train_start
                    total_examples += self.config.get(
                        "static_benchmark.batch_size")
                    batch_id += 1
                    print_step = int(
                        config.get("static_benchmark.print_period"))
                    if batch_id % print_step == 0:
                        profiler_string = ""
                        profiler_string += "avg_batch_cost: {} sec, ".format(
                            format((train_run_cost) / print_step, '.5f'))
                        profiler_string += "avg_samples: {}, ".format(
                            format(total_examples / print_step, '.5f'))
                        profiler_string += "ips: {} {}/sec ".format(
                            format(total_examples / (train_run_cost), '.5f'),
                            self.count_method)
                        logger.info("Epoch: {}, Batch: {}, {}".format(
                            epoch, batch_id, profiler_string))
                        train_run_cost = 0.0
                        total_examples = 0
                except fluid.core.EOFException:
                    self.reader.reset()
                    break


if __name__ == "__main__":
    paddle.enable_static()
    config = parse_args()
    os.environ["CPU_NUM"] = str(config.get("static_benchmark.thread_num"))
    benchmark_main = Main(config)
    benchmark_main.run()
