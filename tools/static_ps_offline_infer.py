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
from utils.static_ps.common import YamlHelper, is_distributed_env, get_utils_file_path
import argparse
import time
import sys
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle
import os
import warnings
import logging
import paddle.fluid as fluid

__dir__ = os.path.dirname(os.path.abspath(__file__))
print(os.path.abspath(os.path.join(__dir__, '..')))

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
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    yaml_helper = YamlHelper()
    config = yaml_helper.load_yaml(args.config_yaml)
    config["yaml_path"] = args.config_yaml
    config["config_abs_dir"] = args.abs_dir
    yaml_helper.print_yaml(config)
    return config


class Main(object):
    def __init__(self, config):
        self.metrics = {}
        self.config = config
        self.input_data = None
        self.reader = None
        self.exe = None
        self.train_result_dict = {}
        self.train_result_dict["speed"] = []

    def run(self):
        fleet.init()
        self.network()
        if fleet.is_server():
            self.run_server()
        elif fleet.is_worker():
            self.run_offline_infer()
            fleet.stop_worker()
            # self.record_result()
        logger.info("Run Success, Exit.")

    def network(self):
        model = get_model(self.config)
        self.input_data = model.create_feeds()
        self.metrics = model.net(self.input_data)
        self.all_vars = model.all_vars
        logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))
        model.create_optimizer(get_strategy(self.config))

    def run_server(self):
        logger.info("Run Server Begin")
        fleet.init_server(config.get("runner.warmup_model_path"))
        fleet.run_server()

    def wait_and_prepare_dataset(self):
        dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_use_var(self.input_data)
        train_data_dir = self.config.get("runner.data_dir", "")
        dataset.set_batch_size(self.config.get('runner.batch_size'))
        dataset.set_thread(self.config.get('runner.thread_num'))
        dataset.set_parse_ins_id(self.config.get("runner.parse_ins_id", False))
        dataset.set_parse_content(
            self.config.get("runner.parse_content", False))

        filelist = []
        # for path in train_data_dir:
        #     filelist += [path + "/%s" % x for x in os.listdir(path)]
        for f in os.listdir(train_data_dir):
            filelist.append("{}/{}".format(train_data_dir, f))
        print("filelist:", filelist)

        dataset.set_filelist(filelist)
        self.pipe_command = "{} {} {}".format(
            self.config.get("runner.pipe_command"),
            config.get("yaml_path"), get_utils_file_path())
        dataset.set_pipe_command(self.pipe_command)
        dataset.load_into_memory()
        return dataset

    def run_offline_infer(self):
        logger.info("Run Offline Infer Begin")
        place = paddle.CPUPlace()
        self.exe = paddle.static.Executor(place)

        self.exe.run(paddle.static.default_startup_program())
        fleet.init_worker()

        init_model_path = config.get("runner.init_model_path")
        fleet.load_model(init_model_path, mode=0)

        logger.info("Prepare Dataset Begin.")
        prepare_data_start_time = time.time()
        dataset = self.wait_and_prepare_dataset()
        prepare_data_end_time = time.time()
        logger.info("Prepare Dataset Done, using time {} second.".format(
            prepare_data_end_time - prepare_data_start_time))

        infer_start_time = time.time()
        self.dataset_offline_infer(dataset)
        infer_end_time = time.time()
        logger.info("Infer Dataset Done, using time {} second.".format(
            infer_end_time - infer_start_time))

    def dataset_offline_infer(self, cur_dataset):
        logger.info("Infer Dataset Begin.")
        fetch_info = ["Var {}".format(var_name) for var_name in self.metrics]
        fetch_vars = [var for _, var in self.metrics.items()]
        print_step = int(config.get("runner.print_interval"))
        dump_fields_path = self.config.get("runner.dump_fields_path")
        dump_fields = [var.name for var in self.all_vars]
        set_dump_config(paddle.static.default_main_program(), {
            "dump_fields_path": dump_fields_path,
            "dump_fields": dump_fields
        })
        print(paddle.static.default_main_program()._fleet_opt)

        self.exe.infer_from_dataset(
            program=paddle.static.default_main_program(),
            dataset=cur_dataset,
            fetch_list=fetch_vars,
            fetch_info=fetch_info,
            print_period=print_step,
            debug=False)
        cur_dataset.release_memory()


if __name__ == "__main__":
    paddle.enable_static()
    config = parse_args()
    # os.environ["CPU_NUM"] = str(config.get("runner.thread_num"))
    benchmark_main = Main(config)
    benchmark_main.run()
