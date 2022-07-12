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
from utils.static_ps.reader_helper import get_reader, get_infer_reader, get_example_num, get_file_list, get_word_num
from utils.static_ps.program_helper import get_model, get_strategy, set_dump_config
from utils.static_ps.common import YamlHelper, is_distributed_env
import argparse
import time
import sys
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
from paddle.distributed.ps.coordinator import FlClient
import paddle
import os
import warnings
import logging
import ast
import numpy as np
import struct

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("PaddleRec train script")
    parser.add_argument(
        '-m',
        '--config_yaml',
        type=str,
        required=True,
        help='config file path')
    parser.add_argument(
        '-bf16',
        '--pure_bf16',
        type=ast.literal_eval,
        default=False,
        help="whether use bf16")
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    yaml_helper = YamlHelper()
    config = yaml_helper.load_yaml(args.config_yaml)
    config["yaml_path"] = args.config_yaml
    config["config_abs_dir"] = args.abs_dir
    config["pure_bf16"] = args.pure_bf16
    yaml_helper.print_yaml(config)
    return config


def bf16_to_fp32(val):
    return np.float32(struct.unpack('<f', struct.pack('<I', val << 16))[0])


class Main(object):
    def __init__(self, config):
        self.metrics = {}
        self.config = config
        self.input_data = None
        self.train_dataset = None
        self.test_dataset = None
        self.exe = None
        self.train_result_dict = {}
        self.train_result_dict["speed"] = []
        self.model = None
        self.pure_bf16 = self.config['pure_bf16']
        self.role = None

    def run(self):
        self.init_fleet_with_gloo()
        self.network()
        if fleet.is_server():
            self.run_server()
        elif fleet.is_worker():
            self.run_worker()
            fleet.stop_worker()
            self.record_result()
        elif fleet.is_coordinator():
            self.run_coordinator()

        logger.info("Run Success, Exit.")

    def init_fleet_with_gloo(self, use_gloo=True):
        if use_gloo:
            os.environ["PADDLE_WITH_GLOO"] = "1"
            self.role = role_maker.PaddleCloudRoleMaker()
            fleet.init(self.role)
        else:
            fleet.init()

    def network(self):
        self.model = get_model(self.config)
        self.input_data = self.model.create_feeds()
        self.inference_feed_var = self.model.create_feeds(is_infer=False)
        self.init_reader()
        self.metrics = self.model.net(self.input_data)
        self.inference_target_var = self.model.inference_target_var
        logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))
        self.model.create_optimizer(get_strategy(self.config))

    def run_coordinator(self):
        logger.info("Run Coordinator Begin")
        fleet.init_coordinator()
        fleet.make_fl_strategy()

    def run_server(self):
        logger.info("Run Server Begin")
        fleet.init_server(config.get("runner.warmup_model_path"))
        fleet.run_server()

    def run_worker(self):
        logger.info("Run Worker Begin")
        use_cuda = int(config.get("runner.use_gpu"))
        place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
        self.exe = paddle.static.Executor(place)

        with open("./{}_worker_main_program.prototxt".format(
                fleet.worker_index()), 'w+') as f:
            f.write(str(paddle.static.default_main_program()))
        with open("./{}_worker_startup_program.prototxt".format(
                fleet.worker_index()), 'w+') as f:
            f.write(str(paddle.static.default_startup_program()))

        self.exe.run(paddle.static.default_startup_program())
        if self.pure_bf16:
            self.model.optimizer.amp_init(self.exe.place)
        fleet.init_worker()
        if self.role is not None:
            self.fl_client = FlClient(self.role)
        else:
            raise ValueError("self.role is none")
        save_model_path = self.config.get("runner.model_save_path")
        if save_model_path and (not os.path.exists(save_model_path)):
            os.makedirs(save_model_path)

        reader_type = self.config.get("runner.reader_type", "QueueDataset")
        epochs = int(self.config.get("runner.epochs"))
        sync_mode = self.config.get("runner.sync_mode")

        if reader_type == "InmemoryDataset":
            self.train_dataset.load_into_memory()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            self.dataset_train_loop(epoch)

            epoch_time = time.time() - epoch_start_time
            epoch_speed = self.example_nums / epoch_time
            logger.info(
                "Epoch: {}, using time {} second, ips {} {}/sec.".format(
                    epoch, epoch_time, epoch_speed, self.count_method))
            self.train_result_dict["speed"].append(epoch_speed)

            model_dir = "{}/{}".format(save_model_path, epoch)
            if fleet.is_first_worker() and save_model_path:
                if is_distributed_env():
                    fleet.save_persistables(self.exe, model_dir)
                else:
                    raise ValueError("it is not distributed env")
            fleet.barrier_worker()

            state_info = {"client id": 0, "auc": 0.9, "epoch": 0}
            self.fl_client.push_fl_client_info_sync(state_info)
            strategy_dict = self.fl_client.pull_fl_strategy()
            print("received fl strategy: {}".format(strategy_dict))
            # ......... to implement ...... #

            self.dataset_online_infer(epoch)

        if reader_type == "InmemoryDataset":
            self.train_dataset.release_memory()

    def init_reader(self):
        if fleet.is_server():
            return
        self.config["runner.reader_type"] = self.config.get(
            "runner.reader_type", "QueueDataset")
        self.train_dataset, self.train_file_list = get_reader(self.input_data,
                                                              config)
        self.test_dataset, self.test_file_list = get_infer_reader(
            self.input_data, config)

        self.example_nums = 0
        self.count_method = self.config.get("runner.example_count_method",
                                            "example")
        if self.count_method == "example":
            self.example_nums = get_example_num(self.train_file_list)
        elif self.count_method == "word":
            self.example_nums = get_word_num(self.train_file_list)
        else:
            raise ValueError(
                "Set static_benchmark.example_count_method for example / word for example count."
            )

    def dataset_online_infer(self, epoch):
        logger.info("Epoch: {}, Running Infer Begin.".format(epoch))
        fetch_info = [
            "Epoch {} Var {}".format(epoch, var_name)
            for var_name in self.metrics
        ]
        fetch_vars = [var for _, var in self.metrics.items()]
        print_step = int(config.get("runner.print_interval"))
        self.exe.infer_from_dataset(
            program=paddle.static.default_main_program(),
            dataset=self.test_dataset,
            fetch_list=fetch_vars,
            fetch_info=fetch_info,
            print_period=print_step,
            debug=False)

    def dataset_train_loop(self, epoch):
        logger.info("Epoch: {}, Running Train Begin.".format(epoch))
        fetch_info = [
            "Epoch {} Var {}".format(epoch, var_name)
            for var_name in self.metrics
        ]
        fetch_vars = [var for _, var in self.metrics.items()]
        print_step = int(config.get("runner.print_interval"))

        debug = config.get("runner.dataset_debug", False)
        if config.get("runner.need_dump"):
            debug = True
            dump_fields_path = "{}/{}".format(
                config.get("runner.dump_fields_path"), epoch)
            set_dump_config(paddle.static.default_main_program(), {
                "dump_fields_path": dump_fields_path,
                "dump_fields": config.get("runner.dump_fields")
            })
        #print(paddle.static.default_main_program()._fleet_opt)
        self.exe.train_from_dataset(
            program=paddle.static.default_main_program(),
            dataset=self.train_dataset,
            fetch_list=fetch_vars,
            fetch_info=fetch_info,
            print_period=print_step,
            debug=debug)

    def record_result(self):
        logger.info("train_result_dict: {}".format(self.train_result_dict))
        with open("./train_result_dict.txt", 'w+') as f:
            f.write(str(self.train_result_dict))


if __name__ == "__main__":
    paddle.enable_static()
    config = parse_args()
    os.environ["CPU_NUM"] = str(config.get("runner.thread_num"))
    benchmark_main = Main(config)
    benchmark_main.run()
