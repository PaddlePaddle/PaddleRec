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
from utils.static_ps.flow_helper import *
from utils.static_ps.metric_helper import get_global_metrics_str, clear_metrics
from utils.static_ps.time_helper import get_avg_cost_mins, get_max_cost_mins, get_min_cost_mins
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
from paddle.distributed.fleet.utils.fs import LocalFS, HDFSClient

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
    config = yaml_helper.load_yaml(args.config_yaml, ["table_parameters"])
    config["yaml_path"] = args.config_yaml
    config["config_abs_dir"] = args.abs_dir
    yaml_helper.print_yaml(config)
    return config


class Main(object):
    def __init__(self, config):
        self.metrics = {}
        self.config = config
        self.exe = None
        self.reader_type = config.get("runner.reader_type", "InMemoryDataset")
        self.split_interval = config.get("runner.split_interval", 5)
        self.split_per_pass = config.get("runner.split_per_pass", 1)
        self.checkpoint_per_pass = config.get("runner.checkpoint_per_pass", 6)
        self.save_delta_frequency = config.get("runner.save_delta_frequency",
                                               6)
        self.save_first_base = config.get("runner.save_first_base", False)
        self.data_donefile = config.get("runner.data_donefile", "")
        self.data_sleep_second = config.get("runner.data_sleep_second", 10)
        self.start_day = config.get("runner.start_day")
        self.end_day = config.get("runner.end_day")
        self.save_model_path = self.config.get("runner.model_save_path")
        self.need_train_dump = self.config.get("runner.need_train_dump", False)
        self.need_infer_dump = self.config.get("runner.need_infer_dump", False)
        if config.get("runner.fs_client.uri") is not None:
            self.hadoop_fs_name = config.get("runner.fs_client.uri", "")
            self.hadoop_fs_ugi = config.get("runner.fs_client.user",
                                            "") + "," + config.get(
                                                "runner.fs_client.passwd", "")
            configs = {
                "fs.default.name": self.hadoop_fs_name,
                "hadoop.job.ugi": self.hadoop_fs_ugi
            }
            self.hadoop_client = HDFSClient("$HADOOP_HOME", configs)
        else:
            self.hadoop_fs_name, self.hadoop_fs_ugi = "", ""
            self.hadoop_client = None

    def run(self):
        self.init_fleet_with_gloo()
        self.init_network()
        if fleet.is_server():
            self.run_server()
        elif fleet.is_worker():
            self.run_online_worker()
            fleet.stop_worker()
            # self.record_result()
        logger.info("Run Success, Exit.")

    def init_fleet_with_gloo(use_gloo=True):
        if use_gloo:
            os.environ["PADDLE_WITH_GLOO"] = "1"
            role = role_maker.PaddleCloudRoleMaker()
            fleet.init(role)
        else:
            fleet.init()

    def init_network(self):
        model = get_model(self.config)
        self.input_data = model.create_feeds()
        self.metrics = model.net(self.input_data)
        self.inference_feed_vars = model.inference_feed_vars
        self.inference_target_var = model.inference_target_var
        if hasattr(model, "all_vars"):
            with open("all_vars.txt", 'w+') as f:
                f.write('\n'.join([var.name for var in model.all_vars]))
        if config.get("runner.need_prune", False):
            # DSSM prune net
            self.inference_feed_vars = self.model.prune_feed_vars
            self.inference_target_var = self.model.prune_target_var
        if config.get("runner.need_train_dump", False):
            self.train_dump_fields = model.train_dump_fields if hasattr(
                self.model, "train_dump_fields") else []
            self.train_dump_params = model.train_dump_params if hasattr(
                self.model, "train_dump_params") else []
        if config.get("runner.need_infer_dump", False):
            self.infer_dump_fields = model.infer_dump_fields if hasattr(
                model, "infer_dump_fields") else []

        self.config['stat_var_names'] = model.thread_stat_var_names
        self.metric_list = model.metric_list
        self.metric_types = model.metric_types

        logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))
        model.create_optimizer(get_strategy(self.config))

    def run_server(self):
        logger.info("Run Server Begin")
        # fleet.init_server(config.get("runner.warmup_model_path", "./warmup"))
        fleet.init_server()
        fleet.run_server()

    def wait_and_prepare_dataset(self, day, pass_index):
        train_data_path = self.config.get("runner.train_data_dir", [])
        dataset = fluid.DatasetFactory().create_dataset(self.reader_type)
        dataset.set_use_var(self.input_data)
        dataset.set_batch_size(self.config.get('runner.train_batch_size', 1))
        dataset.set_thread(self.config.get('runner.train_thread_num', 1))
        dataset.set_hdfs_config(self.hadoop_fs_name, self.hadoop_fs_ugi)
        dataset.set_parse_ins_id(self.config.get("runner.parse_ins_id", False))
        dataset.set_parse_content(
            self.config.get("runner.parse_content", False))

        cur_path = []
        for i in self.online_intervals[pass_index - 1]:
            p = os.path.join(train_data_path, day, str(i))
            if self.data_donefile:
                cur_donefile = os.path.join(p, self.data_donefile)
                data_ready(cur_donefile, self.data_sleep_second,
                           self.hadoop_client)
            cur_path.append(p)
        global_file_list = file_ls(cur_path, self.hadoop_client)
        my_file_list = fleet.util.get_file_shard(global_file_list)
        logger.info("my_file_list = {}".format(my_file_list))
        dataset.set_filelist(my_file_list)

        self.pipe_command = "{} {} {}".format(
            self.config.get("runner.pipe_command"),
            config.get("yaml_path"), get_utils_file_path())
        dataset.set_pipe_command(self.pipe_command)
        dataset.load_into_memory()

        shuffle_thread_num = config.get("runner.shuffle_thread_num", 12)
        begin = time.time()
        dataset.global_shuffle(fleet, shuffle_thread_num)
        end = time.time()
        logger.info('global_shuffle time cost: {}'.format((end - begin) /
                                                          60.0))
        shuffle_data_size = dataset.get_shuffle_data_size(fleet)
        logger.info('after global_shuffle data_size: {}'.format(
            shuffle_data_size))

        return dataset

    def wait_and_prepare_infer_dataset(self, day, pass_index):
        test_data_path = self.config.get("runner.infer_data_dir", [])
        dataset = fluid.DatasetFactory().create_dataset(self.reader_type)
        dataset.set_use_var(self.input_data)
        dataset.set_batch_size(self.config.get('runner.infer_batch_size', 1))
        dataset.set_thread(self.config.get('runner.infer_thread_num', 1))
        dataset.set_hdfs_config(self.hadoop_fs_name, self.hadoop_fs_ugi)
        dataset.set_parse_ins_id(self.config.get("runner.parse_ins_id", False))
        dataset.set_parse_content(
            self.config.get("runner.parse_content", False))

        cur_path = []
        for i in self.online_intervals[pass_index - 1]:
            p = os.path.join(train_data_path, day, str(i))
            if self.data_donefile:
                cur_donefile = os.path.join(p, self.data_donefile)
                data_ready(cur_donefile, self.data_sleep_second,
                           self.hadoop_client)
            cur_path.append(p)
        global_file_list = file_ls(cur_path, self.hadoop_client)
        my_file_list = fleet.util.get_file_shard(global_file_list)
        logger.info("my_file_list = {}".format(my_file_list))
        dataset.set_filelist(my_file_list)

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

        self.online_intervals = get_online_pass_interval(
            self.split_interval, self.split_per_pass, False)
        if is_local(self.save_model_path) and self.save_model_path and (
                not os.path.exists(self.save_model_path)):
            os.makedirs(self.save_model_path)

        last_day, last_pass, last_path, xbox_base_key = get_last_save_model(
            self.save_model_path, self.hadoop_client)
        logger.info(
            "get_last_save_model last_day = {}, last_pass = {}, last_path = {}, xbox_base_key = {}".
            format(last_day, last_pass, last_path, xbox_base_key))
        if last_day != -1 and fleet.is_first_worker():
            load_model(last_path, 0, self.hadoop_client)
        fleet.barrier_worker()

        day = self.start_day
        infer_first = True
        while int(day) <= int(self.end_day):
            logger.info("training a new day {}, end_day = {}".format(
                day, self.end_day))
            if last_day != -1 and int(day) < last_day:
                day = get_next_day(day)
                continue
            # base_model_saved = False
            for pass_id in range(1, 1 + len(self.online_intervals)):
                print(last_day, day, last_pass, pass_id)
                if (last_day != -1 and int(day) == last_day) and (
                        last_pass != -1 and int(pass_id) <= last_pass):
                    continue
                if self.save_first_base and fleet.is_first_worker():
                    self.save_first_base = False
                    last_base_day, last_base_path, tmp_xbox_base_key = \
                        get_last_save_xbox_base(self.save_model_path, self.hadoop_client)
                    logger.info(
                        "get_last_save_xbox_base, last_base_day = {}, last_base_path = {}, tmp_xbox_base_key = {}".
                        format(last_base_day, last_base_path,
                               tmp_xbox_base_key))
                    if int(day) > last_base_day:
                        xbox_base_key = int(time.time())
                        save_xbox_model(self.save_model_path, day, -1,
                                        self.exe, self.inference_feed_vars,
                                        self.inference_target_var,
                                        self.hadoop_client)
                        write_xbox_donefile(
                            output_path=self.save_model_path,
                            day=day,
                            pass_id=-1,
                            xbox_base_key=xbox_base_key,
                            client=self.hadoop_client)
                    elif int(day) == last_base_day:
                        xbox_base_key = tmp_xbox_base_key
                fleet.barrier_worker()

                logger.info("training a new day = {} new pass = {}".format(
                    day, pass_id))
                logger.info("Day:{}, Pass: {}, Prepare Dataset Begin.".format(
                    day, pass_id))
                begin_train = time.time()
                begin = time.time()
                dataset = self.wait_and_prepare_dataset(day, pass_id)
                end = time.time()
                read_data_cost = (end - begin) / 60.0
                logger.info("Prepare Dataset Done, using time {} mins.".format(
                    read_data_cost))

                infer_cost = 0
                infer_metric_cost = 0
                if infer_first:
                    infer_first = False
                else:
                    logger.info("Day:{}, Pass: {}, Infering Dataset Begin.".
                                format(day, pass_id))
                    begin = time.time()
                    self.dataset_infer_loop(dataset, day, pass_id)
                    end = time.time()
                    infer_cost = (end - begin) / 60.0
                    logger.info("Infering Dataset Done, using time {} mins.".
                                format(infer_cost))
                    begin = time.time()
                    metric_str = get_global_metrics_str(fluid.global_scope(),
                                                        self.metric_list, "")
                    logger.info("Day:{}, Pass: {}, Infer Global Metric: {}".
                                format(day, pass_id, metric_str))
                    clear_metrics(fluid.global_scope(), self.metric_list,
                                  self.metric_types)
                    end = time.time()
                    infer_metric_cost = (end - begin) / 60.0

                logger.info("Day:{}, Pass: {}, Training Dataset Begin.".format(
                    day, pass_id))
                begin = time.time()
                self.dataset_train_loop(dataset, day, pass_id,
                                        self.need_train_dump)
                end = time.time()
                avg_cost = get_avg_cost_mins(end - begin)
                get_max_cost_mins(end - begin)
                get_min_cost_mins(end - begin)
                train_cost = avg_cost
                logger.info("Training Dataset Done, using time {} mins.".
                            format(train_cost))

                begin = time.time()
                dataset.release_memory()
                end = time.time()
                release_cost = (end - begin) / 60.0

                begin = time.time()
                metric_str = get_global_metrics_str(fluid.global_scope(),
                                                    self.metric_list, "")
                logger.info("Day:{}, Pass: {}, Train Global Metric: {}".format(
                    day, pass_id, metric_str))
                clear_metrics(fluid.global_scope(), self.metric_list,
                              self.metric_types)
                end = time.time()
                metric_cost = (end - begin) / 60
                end_train = time.time()
                total_cost = (end_train - begin_train) / 60
                other_cost = total_cost - read_data_cost - train_cost - release_cost - metric_cost - infer_cost - infer_metric_cost
                log_str = "finished train epoch %d time cost:%s min job time cost" \
                            ":[read_data:%s min][train: %s min][metric: %s min][release: %s min]" \
                            "[infer:%s min][infer_metric: %s min][other:%s min]" \
                              % (pass_id, total_cost, read_data_cost, train_cost, metric_cost, release_cost, infer_cost, infer_metric_cost, other_cost)
                logger.info(log_str)

                if self.need_infer_dump:
                    prepare_data_start_time = time.time()
                    dump_dataset = self.wait_and_prepare_infer_dataset(day,
                                                                       pass_id)
                    prepare_data_end_time = time.time()
                    logger.info(
                        "Prepare Infer Dump Dataset Done, using time {} second.".
                        format(prepare_data_end_time -
                               prepare_data_start_time))

                    dump_start_time = time.time()
                    self.dataset_infer_loop(dump_dataset, day, pass_id, True)
                    dump_end_time = time.time()
                    logger.info(
                        "Infer Dump Dataset Done, using time {} second.".
                        format(dump_end_time - dump_start_time))

                    dump_dataset.release_memory()

                if fleet.is_first_worker():
                    if pass_id % self.checkpoint_per_pass == 0:
                        save_model(self.exe, self.save_model_path, day,
                                   pass_id)
                        write_model_donefile(
                            output_path=self.save_model_path,
                            day=day,
                            pass_id=pass_id,
                            xbox_base_key=xbox_base_key,
                            client=self.hadoop_client)
                    if pass_id % self.save_delta_frequency == 0:
                        last_xbox_day, last_xbox_pass, last_xbox_path, _ = get_last_save_xbox(
                            self.save_model_path, self.hadoop_client)
                        if int(day) < last_xbox_day or int(
                                day) == last_xbox_day and int(
                                    pass_id) <= last_xbox_pass:
                            log_str = "delta model exists"
                            logger.info(log_str)
                        else:
                            save_xbox_model(self.save_model_path, day, pass_id,
                                            self.exe, self.inference_feed_vars,
                                            self.inference_target_var,
                                            self.hadoop_client)  # 1 delta
                            write_xbox_donefile(
                                output_path=self.save_model_path,
                                day=day,
                                pass_id=pass_id,
                                xbox_base_key=xbox_base_key,
                                client=self.hadoop_client,
                                hadoop_fs_name=self.hadoop_fs_name,
                                monitor_data=metric_str)
                fleet.barrier_worker()

            logger.info("shrink table")
            begin = time.time()
            fleet.shrink()
            end = time.time()
            logger.info("shrink table done, cost %s min" % (
                (end - begin) / 60.0))

            if fleet.is_first_worker():
                last_base_day, last_base_path, last_base_key = get_last_save_xbox_base(
                    self.save_model_path, self.hadoop_client)
                logger.info(
                    "one epoch finishes, get_last_save_xbox, last_base_day = {}, last_base_path = {}, last_base_key = {}".
                    format(last_base_day, last_base_path, last_base_key))
                next_day = get_next_day(day)
                if int(next_day) <= last_base_day:
                    logger.info("batch model/base xbox model exists")
                else:
                    xbox_base_key = int(time.time())
                    save_xbox_model(self.save_model_path, next_day, -1,
                                    self.exe, self.inference_feed_vars,
                                    self.inference_target_var,
                                    self.hadoop_client)
                    write_xbox_donefile(
                        output_path=self.save_model_path,
                        day=next_day,
                        pass_id=-1,
                        xbox_base_key=xbox_base_key,
                        client=self.hadoop_client,
                        hadoop_fs_name=self.hadoop_fs_name,
                        monitor_data=metric_str)
                    save_batch_model(self.exe, self.save_model_path, next_day)
                    write_model_donefile(
                        output_path=self.save_model_path,
                        day=next_day,
                        pass_id=-1,
                        xbox_base_key=xbox_base_key,
                        client=self.hadoop_client)
            fleet.barrier_worker()
            day = get_next_day(day)

    def dataset_train_loop(self, cur_dataset, day, pass_index,
                           need_dump=False):
        fetch_info = [
            "Day: {} Pass: {} Var {}".format(day, pass_index, var_name)
            for var_name in self.metrics
        ]
        fetch_vars = [var for _, var in self.metrics.items()]
        print_step = int(config.get("runner.print_interval"))

        debug = config.get("runner.dataset_debug", False)
        if need_dump:
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

    def dataset_infer_loop(self, cur_dataset, day, pass_index,
                           need_dump=False):
        fetch_info = [
            "Day: {} Pass: {} Var {}".format(day, pass_index, var_name)
            for var_name in self.metrics
        ]
        fetch_vars = [var for _, var in self.metrics.items()]
        print_step = int(config.get("runner.print_interval"))
        debug = config.get("runner.dataset_debug", False)
        if need_dump:
            dump_fields_dir = self.config.get("runner.infer_dump_fields_dir")
            dump_fields_path = "{}/{}/{}".format(dump_fields_dir, day,
                                                 pass_index)
            dump_fields = [var.name for var in self.infer_dump_fields]
            set_dump_config(paddle.static.default_main_program(), {
                "dump_fields_path": dump_fields_path,
                "dump_fields": dump_fields
            })

        self.exe.infer_from_dataset(
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
            })


if __name__ == "__main__":
    paddle.enable_static()
    config = parse_args()
    # os.environ["CPU_NUM"] = str(config.get("runner.thread_num"))
    benchmark_main = Main(config)
    benchmark_main.run()
