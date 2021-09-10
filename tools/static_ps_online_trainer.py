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
            self.run_online_worker()
            fleet.stop_worker()
            # self.record_result()
        logger.info("Run Success, Exit.")

    def network(self):
        model = get_model(self.config)
        self.input_data = model.create_feeds()
        self.inference_feed_vars = model.create_feeds(is_infer=True)
        self.metrics = model.net(self.input_data)
        self.inference_target_var = model.inference_target_var
        if config.get("runner.need_prune", False):
            # DSSM prune net
            self.inference_feed_vars = model.prune_feed_vars
            self.inference_target_var = model.prune_target_var
        if config.get("runner.need_train_dump", False):
            self.train_dump_fields = model.train_dump_fields if hasattr(
                model, "train_dump_fields") else []
            self.train_dump_params = model.train_dump_params if hasattr(
                model, "train_dump_params") else []
        if config.get("runner.need_infer_dump", False):
            self.infer_dump_fields = model.infer_dump_fields if hasattr(
                model, "infer_dump_fields") else []
        print(self.inference_feed_vars)
        print(self.inference_target_var)
        logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))
        model.create_optimizer(get_strategy(self.config))

    def run_server(self):
        logger.info("Run Server Begin")
        fleet.init_server(config.get("runner.warmup_model_path"))
        fleet.run_server()

    def wait_and_prepare_dataset(self, day, pass_index):
        train_data_dir = self.config.get("runner.train_data_dir", [])

        dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_use_var(self.input_data)
        dataset.set_batch_size(self.config.get('runner.train_batch_size'))
        dataset.set_thread(self.config.get('runner.train_thread_num'))
        dataset.set_input_type(self.config.get('runner.input_type', 0))
        dataset.set_hdfs_config(
            self.config.get('runner.fs_name'),
            self.config.get('runner.fs_ugi'))
        dataset.set_parse_ins_id(self.config.get("runner.parse_ins_id", False))
        dataset.set_parse_content(
            self.config.get("runner.parse_content", False))

        # may you need define your dataset_filelist for day/pass_index
        train_data_dir = "{}/{}/{}".format(train_data_dir, day, pass_index)
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

    def wait_and_prepare_infer_dataset(self, day, pass_index):
        test_data_dir = self.config.get("runner.test_data_dir", [])
        dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_use_var(self.input_data)
        dataset.set_batch_size(self.config.get('runner.test_batch_size'))
        dataset.set_thread(self.config.get('runner.test_thread_num'))
        dataset.set_input_type(self.config.get('runner.input_type', 0))
        dataset.set_hdfs_config(
            self.config.get('runner.fs_name'),
            self.config.get('runner.fs_ugi'))
        dataset.set_parse_ins_id(self.config.get("runner.parse_ins_id", False))
        dataset.set_parse_content(
            self.config.get("runner.parse_content", False))

        # may you need define your dataset_filelist for day/pass_index
        test_data_dir = "{}/{}/{}".format(test_data_dir, day, pass_index)
        filelist = []
        # for path in test_data_dir:
        #     filelist += [path + "/%s" % x for x in os.listdir(path)]
        for f in os.listdir(test_data_dir):
            filelist.append("{}/{}".format(test_data_dir, f))
        print("filelist:", filelist)

        dataset.set_filelist(filelist)
        self.pipe_command = "{} {} {}".format(
            self.config.get("runner.pipe_command"),
            config.get("yaml_path"), get_utils_file_path())
        dataset.set_pipe_command(self.pipe_command)
        dataset.load_into_memory()
        return dataset

    def run_online_worker(self):
        logger.info("Run Online Worker Begin")
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
        fleet.init_worker()

        save_model_path = self.config.get("runner.model_save_path")
        if save_model_path and (not os.path.exists(save_model_path)):
            os.makedirs(save_model_path)

        days = os.popen("echo -n " + self.config.get("runner.days")).read(
        ).split(" ")
        pass_per_day = int(self.config.get("runner.pass_per_day"))

        for day_index in range(len(days)):
            day = days[day_index]
            for pass_index in range(1, pass_per_day + 1):
                logger.info("Day: {} Pass: {} Begin.".format(day, pass_index))

                prepare_data_start_time = time.time()
                dataset = self.wait_and_prepare_dataset(day, pass_index)
                prepare_data_end_time = time.time()
                logger.info(
                    "Prepare Dataset Done, using time {} second.".format(
                        prepare_data_end_time - prepare_data_start_time))

                train_start_time = time.time()
                self.dataset_train_loop(dataset, day, pass_index)
                train_end_time = time.time()
                logger.info("Train Dataset Done, using time {} second.".format(
                    train_end_time - train_start_time))

                need_infer_dump = self.config.get("runner.need_infer_dump",
                                                  False)
                if need_infer_dump:
                    prepare_data_start_time = time.time()
                    dump_dataset = self.wait_and_prepare_infer_dataset(
                        day, pass_index)
                    prepare_data_end_time = time.time()
                    logger.info(
                        "Prepare Infer Dump Dataset Done, using time {} second.".
                        format(prepare_data_end_time -
                               prepare_data_start_time))

                    dump_start_time = time.time()
                    self.dataset_dump_loop(dump_dataset, day, pass_index)
                    dump_end_time = time.time()
                    logger.info(
                        "Infer Dump Dataset Done, using time {} second.".
                        format(dump_end_time - dump_start_time))

                model_dir = "{}/{}/{}".format(save_model_path, day, pass_index)

                if fleet.is_first_worker(
                ) and save_model_path and is_distributed_env():
                    fleet.save_inference_model(
                        self.exe,
                        model_dir,
                        [feed.name for feed in self.inference_feed_vars],
                        self.inference_target_var,
                        mode=2)

            if fleet.is_first_worker(
            ) and save_model_path and is_distributed_env():
                fleet.save_inference_model(
                    self.exe,
                    model_dir,
                    [feed.name for feed in self.inference_feed_vars],
                    self.inference_target_var,
                    mode=0)

    def dataset_train_loop(self, cur_dataset, day, pass_index):
        logger.info("Day: {} Pass: {}, Running Dataset Begin.".format(
            day, pass_index))
        fetch_info = [
            "Day: {} Pass: {} Var {}".format(day, pass_index, var_name)
            for var_name in self.metrics
        ]
        fetch_vars = [var for _, var in self.metrics.items()]
        print_step = int(config.get("runner.print_interval"))

        debug = config.get("runner.dataset_debug", False)
        need_dump = config.get("runner.need_train_dump", False)
        if need_dump:
            debug = True
            dump_fields_dir = self.config.get("runner.train_dump_fields_dir")
            dump_fields_path = "{}/{}/{}".format(dump_fields_dir, day,
                                                 pass_index)
            dump_fields = [var.name for var in self.train_dump_fields]
            dump_params = [param.name for param in self.train_dump_params]
            set_dump_config(paddle.static.default_main_program(), {
                "dump_fields_path": dump_fields_path,
                "dump_fields": dump_fields,
                "dump_param": dump_params
            })
        print(paddle.static.default_main_program()._fleet_opt)

        self.exe.train_from_dataset(
            program=paddle.static.default_main_program(),
            dataset=cur_dataset,
            fetch_list=fetch_vars,
            fetch_info=fetch_info,
            print_period=print_step,
            debug=debug)

        if need_dump:
            set_dump_config(paddle.static.default_main_program(), {
                "dump_fields_path": "",
                "dump_fields": [],
                "dump_param": []
            })
        cur_dataset.release_memory()

    def dataset_dump_loop(self, cur_dataset, day, pass_index):
        logger.info("Day: {} Pass: {}, Dump Dataset Begin.".format(day,
                                                                   pass_index))
        fetch_info = [
            "Day: {} Pass: {} Var {}".format(day, pass_index, var_name)
            for var_name in self.metrics
        ]
        fetch_vars = [var for _, var in self.metrics.items()]
        print_step = int(config.get("runner.print_interval"))
        dump_fields_dir = self.config.get("runner.infer_dump_fields_dir")
        dump_fields_path = "{}/{}/{}".format(dump_fields_dir, day, pass_index)
        dump_fields = [var.name for var in self.infer_dump_fields]
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
            debug=True)

        set_dump_config(paddle.static.default_main_program(), {
            "dump_fields_path": "",
            "dump_fields": [],
        })
        cur_dataset.release_memory()


if __name__ == "__main__":
    paddle.enable_static()
    config = parse_args()
    # os.environ["CPU_NUM"] = str(config.get("runner.thread_num"))
    benchmark_main = Main(config)
    benchmark_main.run()
