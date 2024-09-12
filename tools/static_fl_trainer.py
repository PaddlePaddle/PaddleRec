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
os.environ['FLAGS_enable_pir_api'] = '0'
from utils.static_ps.reader_helper import get_reader, get_infer_reader, get_example_num, get_file_list, get_word_num
from utils.static_ps.program_helper import get_model, get_strategy, set_dump_config
from utils.static_ps.common_ps import YamlHelper, is_distributed_env
import argparse
import time
import sys
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
from paddle.distributed.ps.coordinator import FLClient
import paddle
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


class MyFLClient(FLClient):
    def __init__(self):
        pass


class Trainer(object):
    def __init__(self, config):
        self.metrics = {}
        self.config = config
        self.input_data = None
        self.train_dataset = None
        self.test_dataset = None
        self.model = None
        self.pure_bf16 = self.config['pure_bf16']
        self.use_cuda = int(self.config.get("runner.use_gpu"))
        self.place = paddle.CUDAPlace(0) if self.use_cuda else paddle.CPUPlace(
        )
        self.role = None

    def run(self):
        self.init_fleet()
        self.init_network()
        if fleet.is_server():
            self.run_server()
        elif fleet.is_worker():
            self.init_reader()
            self.run_worker()
        elif fleet.is_coordinator():
            self.run_coordinator()

        logger.info("Run Success, Exit.")

    def init_fleet(self, use_gloo=True):
        if use_gloo:
            os.environ["PADDLE_WITH_GLOO"] = "1"
            self.role = role_maker.PaddleCloudRoleMaker()
            fleet.init(self.role)
        else:
            fleet.init()

    def init_network(self):
        self.model = get_model(self.config)
        self.input_data = self.model.create_feeds()
        self.metrics = self.model.net(self.input_data)
        self.model.create_optimizer(get_strategy(self.config))  ## get_strategy
        if self.pure_bf16:
            self.model.optimizer.amp_init(self.place)

    def init_reader(self):
        self.train_dataset, self.train_file_list = get_reader(self.input_data,
                                                              config)
        self.test_dataset, self.test_file_list = get_infer_reader(
            self.input_data, config)

        if self.role is not None:
            self.fl_client = MyFLClient()
            self.fl_client.set_basic_config(self.role, self.config,
                                            self.metrics)
        else:
            raise ValueError("self.role is none")

        self.fl_client.set_train_dataset_info(self.train_dataset,
                                              self.train_file_list)
        self.fl_client.set_test_dataset_info(self.test_dataset,
                                             self.test_file_list)

        example_nums = 0
        self.count_method = self.config.get("runner.example_count_method",
                                            "example")
        if self.count_method == "example":
            example_nums = get_example_num(self.train_file_list)
        elif self.count_method == "word":
            example_nums = get_word_num(self.train_file_list)
        else:
            raise ValueError(
                "Set static_benchmark.example_count_method for example / word for example count."
            )
        self.fl_client.set_train_example_num(example_nums)

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
        self.fl_client.run()


if __name__ == "__main__":
    paddle.enable_static()
    config = parse_args()
    os.environ["CPU_NUM"] = str(config.get("runner.thread_num"))
    trainer = Trainer(config)
    trainer.run()
