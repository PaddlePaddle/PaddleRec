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
from utils.static_ps.program_helper import get_model, get_strategy
from utils.static_ps.common import YamlHelper, is_distributed_env
from utils.utils_single import auc
import argparse
import time
import sys
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle
import os
import warnings
import logging
from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
fleet_util = FleetUtil()

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
        self.model = None
        self.PSGPU = None
        self.train_result_dict = {}
        self.train_result_dict["speed"] = []
        self.train_result_dict["auc"] = []

    def run(self):
        fleet.init()
        self.network()
        if fleet.is_server():
            self.run_server()
        elif fleet.is_worker():
            self.run_worker()
            fleet.stop_worker()
            self.record_result()
        logger.info("Run Success, Exit.")
        logger.info("-" * 100)

    def network(self):
        self.model = get_model(self.config)
        self.input_data = self.model.create_feeds()
        self.init_reader()
        self.metrics = self.model.net(self.input_data)
        self.inference_target_var = self.model.inference_target_var
        logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))
        self.model.create_optimizer(get_strategy(self.config))
        logger.info("end network.....")

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
        fleet.init_worker()

        save_model_path = self.config.get("runner.model_save_path")
        if save_model_path and (not os.path.exists(save_model_path)):
            os.makedirs(save_model_path)

        reader_type = self.config.get("runner.reader_type", None)
        epochs = int(self.config.get("runner.epochs"))
        sync_mode = self.config.get("runner.sync_mode")

        gpus_env = os.getenv("FLAGS_selected_gpus")
        self.PSGPU = paddle.fluid.core.PSGPU()
        gpuslot = [int(i) for i in range(1, self.model.sparse_inputs_slots)]
        self.PSGPU.set_slot_vector(gpuslot)
        self.PSGPU.init_gpu_ps([int(s) for s in gpus_env.split(",")])
        opt_info = paddle.fluid.default_main_program()._fleet_opt
        if use_auc is True:
            opt_info['stat_var_names'] = [
                self.model.stat_pos.name, self.model.stat_neg.name
            ]
        else:
            opt_info['stat_var_names'] = []

        for epoch in range(epochs):
            epoch_start_time = time.time()

            if sync_mode == "heter":
                self.heter_train_loop(epoch)
            elif sync_mode == "gpubox":
                self.reader._set_use_ps_gpu(1)
                self.dataset_train_loop(epoch)
            elif reader_type == "QueueDataset":
                self.dataset_train_loop(epoch)
            elif reader_type == "DataLoader":
                self.dataloader_train_loop(epoch)
            elif reader_type == None or reader_type == "RecDataset":
                self.recdataset_train_loop(epoch)

            epoch_time = time.time() - epoch_start_time
            epoch_speed = self.example_nums / epoch_time
            if use_auc is True:
                global_auc = auc(self.model.stat_pos, self.model.stat_neg,
                                 paddle.fluid.global_scope(), fleet.util)
                self.train_result_dict["auc"].append(global_auc)
                fleet_util.set_zero(self.model.stat_pos.name,
                                    paddle.fluid.global_scope())
                fleet_util.set_zero(self.model.stat_neg.name,
                                    paddle.fluid.global_scope())
                fleet_util.set_zero(self.model.batch_stat_pos.name,
                                    paddle.fluid.global_scope())
                fleet_util.set_zero(self.model.batch_stat_neg.name,
                                    paddle.fluid.global_scope())
                logger.info(
                    "Epoch: {}, using time {} second, ips {} {}/sec. auc: {}".
                    format(epoch, epoch_time, epoch_speed, self.count_method,
                           global_auc))
            else:
                logger.info(
                    "Epoch: {}, using time {} second, ips {} {}/sec.".format(
                        epoch, epoch_time, epoch_speed, self.count_method))
            self.train_result_dict["speed"].append(epoch_speed)

            model_dir = "{}/{}".format(save_model_path, epoch)
            if fleet.is_first_worker(
            ) and save_model_path and is_distributed_env():
                fleet.save_inference_model(
                    self.exe, model_dir,
                    [feed.name for feed in self.input_data],
                    self.inference_target_var)
            fleet.barrier_worker()
            self.reader.release_memory()
            self.PSGPU.end_pass()
            logger.info("finish {} epoch training....".format(epoch))
        self.PSGPU.finalize()

    def init_reader(self):
        if fleet.is_server():
            return
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
        start_time = time.time()
        self.reader.load_into_memory()
        print("self.reader.load_into_memory cost :{} seconds".format(time.time(
        ) - start_time))

        begin_pass_time = time.time()
        self.PSGPU.begin_pass()
        print("begin_pass cost:{} seconds".format(time.time() -
                                                  begin_pass_time))

        logger.info("Epoch: {}, Running Dataset Begin.".format(epoch))
        fetch_info = [
            "Epoch {} Var {}".format(epoch, var_name)
            for var_name in self.metrics
        ]
        fetch_vars = [var for _, var in self.metrics.items()]
        print_step = int(config.get("runner.print_interval"))
        self.exe.train_from_dataset(
            program=paddle.static.default_main_program(),
            dataset=self.reader,
            debug=config.get("runner.dataset_debug"))

    def dataloader_train_loop(self, epoch):
        logger.info("Epoch: {}, Running DataLoader Begin.".format(epoch))
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
                total_examples += (self.config.get("runner.train_batch_size"))
                batch_id += 1
                print_step = int(config.get("runner.print_interval"))
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
            except paddle.fluid.core.EOFException:
                self.reader.reset()
                break

    def recdataset_train_loop(self, epoch):
        logger.info("Epoch: {}, Running RecDatast Begin.".format(epoch))

        input_data_names = [var.name for var in self.input_data]
        batch_size = config.get("runner.train_batch_size", None)
        print_interval = config.get("runner.print_interval", None)

        batch_id = 0
        train_run_cost = 0.0
        train_reader_cost = 0.0
        total_samples = 0
        reader_start = time.time()
        for batch_id, batch_data in enumerate(self.reader()):
            train_reader_cost += time.time() - reader_start
            train_start = time.time()
            # --------------------------------------------------- #
            fetch_batch_var = self.exe.run(
                program=paddle.static.default_main_program(),
                feed=dict(zip(input_data_names, batch_data)),
                fetch_list=[var for _, var in self.metrics.items()])
            # --------------------------------------------------- #
            train_run_cost += time.time() - train_start
            total_samples += batch_size
            if batch_id % print_interval == 0:
                metric_str = ""
                for var_idx, var_name in enumerate(self.metrics):
                    metric_str += "{}: {}, ".format(var_name,
                                                    fetch_batch_var[var_idx])
                logger.info(
                    "Epoch: {}, Batch_id: {}, ".format(epoch,
                                                       batch_id) + metric_str +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} {}/sec".
                    format(train_reader_cost / print_interval, (
                        train_reader_cost + train_run_cost) / print_interval,
                           total_samples / print_interval, total_samples / (
                               train_reader_cost + train_run_cost),
                           self.count_method))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()

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
