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
from utils.static_ps.reader_helper import get_reader, get_example_num, get_file_list, get_word_num
from utils.static_ps.program_helper import get_model, get_strategy, set_dump_config
from utils.static_ps.metric_helper import set_zero, get_global_auc
from utils.static_ps.common_ps import YamlHelper, is_distributed_env
import argparse
import time
import sys
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle
import os
import warnings
import logging
import ast
import numpy as np
import struct
from utils.utils_single import auc

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("PaddleRec train script")
    parser.add_argument("-o", "--opt", nargs='*', type=str)
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
    # modify config from command
    if args.opt:
        for parameter in args.opt:
            parameter = parameter.strip()
            key, value = parameter.split("=")
            if type(config.get(key)) is int:
                value = int(value)
            if type(config.get(key)) is float:
                value = float(value)
            if type(config.get(key)) is bool:
                value = (True if value.lower() == "true" else False)
            config[key] = value
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
        self.reader = None
        self.exe = None
        self.train_result_dict = {}
        self.train_result_dict["speed"] = []
        self.train_result_dict["auc"] = []
        self.model = None
        self.pure_bf16 = self.config['pure_bf16']

    def run(self):
        self.init_fleet_with_gloo()
        self.network()
        if fleet.is_server():
            self.run_server()
        elif fleet.is_worker():
            self.run_worker()
            fleet.stop_worker()
            self.record_result()
        logger.info("Run Success, Exit.")

    def init_fleet_with_gloo(use_gloo=True):
        if use_gloo:
            os.environ["PADDLE_WITH_GLOO"] = "1"
            role = role_maker.PaddleCloudRoleMaker()
            fleet.init(role)
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

    def run_server(self):
        logger.info("Run Server Begin")
        fleet.init_server(config.get("runner.warmup_model_path"))
        fleet.run_server()

    def run_worker(self):
        logger.info("Run Worker Begin")
        use_cuda = int(config.get("runner.use_gpu"))
        use_auc = config.get("runner.use_auc", False)
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

        save_model_path = self.config.get("runner.model_save_path")
        if save_model_path and (not os.path.exists(save_model_path)):
            os.makedirs(save_model_path)

        reader_type = self.config.get("runner.reader_type", "QueueDataset")
        epochs = int(self.config.get("runner.epochs"))
        sync_mode = self.config.get("runner.sync_mode")
        opt_info = paddle.static.default_main_program()._fleet_opt
        if use_auc is True:
            opt_info['stat_var_names'] = [
                self.model.stat_pos.name, self.model.stat_neg.name
            ]
        else:
            opt_info['stat_var_names'] = []

        if reader_type == "InmemoryDataset":
            self.reader.load_into_memory()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            if sync_mode == "heter":
                self.heter_train_loop(epoch)
            elif reader_type == "QueueDataset":
                self.dataset_train_loop(epoch)
            elif reader_type == "InmemoryDataset":
                self.dataset_train_loop(epoch)

            epoch_time = time.time() - epoch_start_time
            epoch_speed = self.example_nums / epoch_time
            if use_auc is True:
                global_auc = get_global_auc(paddle.static.global_scope(),
                                            self.model.stat_pos.name,
                                            self.model.stat_neg.name)
                self.train_result_dict["auc"].append(global_auc)
                set_zero(self.model.stat_pos.name,
                         paddle.static.global_scope())
                set_zero(self.model.stat_neg.name,
                         paddle.static.global_scope())
                set_zero(self.model.batch_stat_pos.name,
                         paddle.static.global_scope())
                set_zero(self.model.batch_stat_neg.name,
                         paddle.static.global_scope())
                logger.info(
                    "Epoch: {}, using time: {} second, ips: {} {}/sec. auc: {}".
                    format(epoch, epoch_time, epoch_speed, self.count_method,
                           global_auc))
            else:
                logger.info(
                    "Epoch: {}, using time {} second, ips {} {}/sec.".format(
                        epoch, epoch_time, epoch_speed, self.count_method))

            self.train_result_dict["speed"].append(epoch_speed)

            model_dir = "{}/{}".format(save_model_path, epoch)

            if is_distributed_env():
                fleet.save_inference_model(
                    self.exe, model_dir,
                    [feed.name for feed in self.inference_feed_var],
                    self.inference_target_var)
            else:
                paddle.static.save_inference_model(
                    model_dir,
                    [feed.name for feed in self.inference_feed_var],
                    [self.inference_target_var], self.exe)

        if reader_type == "InmemoryDataset":
            self.reader.release_memory()

    def init_reader(self):
        if fleet.is_server():
            return
        self.config["runner.reader_type"] = self.config.get(
            "runner.reader_type", "QueueDataset")
        self.reader, self.file_list = get_reader(self.input_data, config)
        self.example_nums = 0
        self.count_method = self.config.get("runner.example_count_method",
                                            "example")
        if self.count_method == "example":
            self.example_nums = get_example_num(self.file_list)
        elif self.count_method == "word":
            self.example_nums = get_word_num(self.file_list)
        else:
            raise ValueError(
                "Set static_benchmark.example_count_method for example / word for example count."
            )

    def dataset_train_loop(self, epoch):
        logger.info("Epoch: {}, Running Dataset Begin.".format(epoch))
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
        print(paddle.static.default_main_program()._fleet_opt)
        self.exe.train_from_dataset(
            program=paddle.static.default_main_program(),
            dataset=self.reader,
            fetch_list=fetch_vars,
            fetch_info=fetch_info,
            print_period=print_step,
            debug=debug)

    def heter_train_loop(self, epoch):
        logger.info(
            "Epoch: {}, Running Begin. Check running metrics at heter_log".
            format(epoch))
        reader_type = self.config.get("runner.reader_type")
        if reader_type == "QueueDataset":
            self.exe.train_from_dataset(
                program=paddle.static.default_main_program(),
                dataset=self.reader,
                debug=config.get("runner.dataset_debug"))
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
                    total_examples += self.config.get("runner.batch_size")
                    batch_id += 1
                    print_step = int(config.get("runner.print_period"))
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
                except paddle.core.EOFException:
                    self.reader.reset()
                    break

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
