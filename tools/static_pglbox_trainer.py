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
from utils.static_ps.config_fleet import get_strategy
from utils.static_ps.common_ps import YamlHelper, is_distributed_env
from utils.static_ps.graph import DistGraph
from utils.static_ps import util
from utils.static_ps.embedding import DistEmbedding
from utils.static_ps.dataset import UnsupReprLearningDataset, InferDataset
from utils.util_config import prepare_config, pretty
from datetime import datetime, timedelta

from pgl.utils.logger import log
from utils.place import get_cuda_places
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

from utils.static_ps.distributed_program import make_distributed_train_program, make_distributed_infer_program
import profiler

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
print(os.path.abspath(os.path.join(__dir__, '..')))

import models.graph.models as Model
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
        '--profiler_options',
        type=str,
        default=None,
        help='The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\".'
    )
    args = parser.parse_args()

    config = prepare_config(args.config_yaml)
    config.local_model_path = "./model"
    config.local_result_path = "./embedding"
    config.local_dump_path = "./dump_walk"
    config.model_save_path = os.path.join(config.working_root, "model")
    config.infer_result_path = os.path.join(config.working_root, 'embedding')
    config.dump_save_path = os.path.join(config.working_root, 'dump_walk')
    config.max_steps = config.max_steps if config.max_steps else 0
    config.metapath_split_opt = config.metapath_split_opt \
                                if config.metapath_split_opt else False
    if args.opt:
        for parameter in args.opt:
            parameter = parameter.strip()
            key, value = parameter.split("=")
            value = int(value)
            config.__setattr__(key, value)

    config.profiler_options = args.profiler_options
    print("#===================PRETTY CONFIG============================#")
    pretty(config, indent=0)
    return config


def get_dataset(inputs, config):
    sage_mode = config.sage_mode if config.sage_mode else False
    fs_name = config.fs_name if config.fs_name is not None else ""
    fs_ugi = config.fs_ugi if config.fs_ugi is not None else ""

    str_samples = util.sample_list_to_str(sage_mode, config.samples)
    str_infer_samples = util.sample_list_to_str(sage_mode,
                                                config.infer_samples)

    excluded_train_pair = config.excluded_train_pair if config.excluded_train_pair else ""
    infer_node_type = config.infer_node_type if config.infer_node_type else ""

    uniq_factor = 0.4
    if not sage_mode:
        train_pass_cap = int(config.walk_len * config.walk_times * config.sample_times_one_chunk \
                         * config.batch_size * uniq_factor)
    else:
        # If sage_mode is True, self.samples can not be None.
        train_pass_cap = int(config.walk_len * config.walk_times * config.sample_times_one_chunk \
                         * config.batch_size * uniq_factor * config.samples[0])

    infer_pass_cap = 10000000  # 1kw
    if config.train_pass_cap:
        train_pass_cap = config.train_pass_cap
    if config.infer_pass_cap:
        infer_pass_cap = config.infer_pass_cap

    get_degree = sage_mode and (config.use_degree_norm
                                if config.use_degree_norm else False)
    graph_config = {
        "walk_len": config.walk_len,
        "walk_degree": config.walk_times,
        "once_sample_startid_len": config.batch_size,
        "sample_times_one_chunk": config.sample_times_one_chunk,
        "window": config.win_size,
        "debug_mode": config.debug_mode,
        "batch_size": config.batch_size,
        "meta_path": config.meta_path,
        "gpu_graph_training": not self.is_predict,
        "sage_mode": sage_mode,
        "samples": str_samples,
        "train_table_cap": train_pass_cap,
        "infer_table_cap": infer_pass_cap,
        "excluded_train_pair": excluded_train_pair,
        "infer_node_type": infer_node_type,
        "get_degree": get_degree
    }

    dataset = paddle.distributed.InMemoryDataset()
    dataset.set_feed_type("SlotRecordInMemoryDataFeed")
    dataset._set_use_ps_gpu(config.get('runner.use_gpu'))
    pipe_cmd = config.get('runner.pipe_command')
    dataset.init(
        use_var=inputs,
        pipe_command=pipe_cmd,
        batch_size=1,
        thread_num=int(config.get('runner.thread_num')),
        fs_name=config.get("runner.fs_name", ""),
        fs_ugi=config.get("runner.fs_ugi", ""), )
    file_list = self.get_file_list(chunk_index)
    dataset.set_filelist(file_list)
    dataset.update_settings(
        parse_ins_id=config.get("runner.parse_ins_id", False),
        parse_content=config.get("runner.parse_content", False), )

    return dataset


def get_file_list(self, chunk_index):
    """ get data file list """
    work_dir = "./workdir/filelist"  # a tmp directory that does not used in other places

    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)

    file_list = []
    chunk_num = 10
    for thread_id in range(len(get_cuda_places())):
        filename = os.path.join(work_dir, "%s_%s_%s" %
                                (chunk_num, chunk_index, thread_id))
        file_list.append(filename)
        with open(filename, "w") as writer:
            writer.write("%s_%s_%s\n" % (chunk_num, chunk_index, thread_id))
    return file_list


class Main(object):
    def __init__(self, config):
        self.metrics = {}
        self.config = config
        self.profiler_options = config.profiler_options
        self.input_data = None
        self.reader = None
        self.exe = None
        self.model = None
        self.train_result_dict = {}
        self.train_result_dict["speed"] = []
        self.train_result_dict["auc"] = []

    def run(self):
        device_ids = get_cuda_places()
        place = paddle.CUDAPlace(device_ids[0])
        self.exe = paddle.static.Executor(place)
        fleet.init()
        self.network()
        if fleet.is_server():
            self.run_server()
        elif fleet.is_worker():
            ret = self.run_worker()
            if ret != 0:
                fleet.stop_worker()
                return -1
        fleet.stop_worker()
        self.record_result()
        logger.info("Run Success, Exit.")
        logger.info("-" * 100)

    def network(self):
        startup_program = paddle.static.Program()
        train_program = paddle.static.Program()
        self.infer_model_dict = None
        with paddle.static.program_guard(train_program, startup_program):
            with paddle.utils.unique_name.guard():
                self.model_dict = getattr(Model, self.config.model_type)(
                    config=self.config)

        self.model_dict.startup_program = startup_program
        self.model_dict.train_program = train_program

        adam = paddle.optimizer.Adam(learning_rate=self.config.dense_lr)
        optimizer = fleet.distributed_optimizer(
            adam, strategy=get_strategy(self.config, self.model_dict))
        optimizer.minimize(self.model_dict.loss,
                           self.model_dict.startup_program)
        make_distributed_train_program(self.config, self.model_dict)

        if self.config.need_inference:
            infer_startup_program = paddle.static.Program()
            infer_train_program = paddle.static.Program()

            with paddle.static.program_guard(infer_train_program,
                                             infer_startup_program):
                with paddle.utils.unique_name.guard():
                    self.infer_model_dict = getattr(
                        Model, self.config.model_type)(config=self.config,
                                                       is_predict=True)

            self.infer_model_dict.startup_program = infer_startup_program
            self.infer_model_dict.train_program = infer_train_program

            fake_lr_infer = 0.00
            adam_infer = paddle.optimizer.Adam(learning_rate=fake_lr_infer)
            optimizer1 = fleet.distributed_optimizer(
                adam_infer,
                strategy=get_strategy(self.config, self.infer_model_dict))
            optimizer1.minimize(self.infer_model_dict.loss,
                                self.infer_model_dict.startup_program)
            make_distributed_infer_program(self.config, self.infer_model_dict)

        logger.info("end network.....")

    def run_server(self):
        logger.info("Run Server Begin")
        fleet.init_server()
        fleet.run_server()

    def run_worker(self):
        logger.info("Run Worker Begin")

        with open("./{}_worker_main_program.prototxt".format(
                fleet.worker_index()), 'w+') as f:
            f.write(str(paddle.static.default_main_program()))
        with open("./{}_worker_startup_program.prototxt".format(
                fleet.worker_index()), 'w+') as f:
            f.write(str(paddle.static.default_startup_program()))

        self.exe.run(self.model_dict.startup_program)
        if self.config.need_inference:
            self.exe.run(self.infer_model_dict.startup_program)
        fleet.init_worker()
        slot_num_for_pull_feature = 1 if self.config.token_slot else 0
        slot_num_for_pull_feature += len(self.config.slots)
        embedding = DistEmbedding(
            slots=self.model_dict.total_gpups_slots,
            embedding_size=self.config.emb_size,
            slot_num_for_pull_feature=slot_num_for_pull_feature)
        dist_graph = DistGraph(
            root_dir=self.config.graph_data_local_path,
            node_types=self.config.ntype2files,
            edge_types=self.config.etype2files,
            symmetry=self.config.symmetry,
            slots=self.config.slots,
            token_slot=self.config.token_slot,
            slot_num_for_pull_feature=slot_num_for_pull_feature,
            num_parts=self.config.num_part,
            metapath_split_opt=self.config.metapath_split_opt)

        dist_graph.load_edge()
        ret = dist_graph.load_node()
        if ret != 0:
            return -1

        fleet.barrier_worker()
        train_dataset = UnsupReprLearningDataset(
            self.config.chunk_num,
            dataset_config=self.config,
            holder_list=self.model_dict.holder_list,
            embedding=embedding,
            dist_graph=dist_graph)

        infer_dataset = InferDataset(
            self.config.chunk_num,
            dataset_config=self.config,
            holder_list=self.model_dict.holder_list,
            infer_model_dict=self.infer_model_dict,
            embedding=embedding,
            dist_graph=dist_graph)

        if self.config.need_train:
            self.train(train_dataset)
        else:
            log.info("STAGE: need_train is %s, skip training process" %
                     self.config.need_train)

        fleet.barrier_worker()
        if self.config.need_inference:
            self.infer(infer_dataset)
        else:
            log.info("STAGE: need_inference is %s, skip inference process" %
                     self.config.need_inference)

    def init_reader(self):
        if fleet.is_server():
            return
        fleet.barrier_work()
        self.reader = get_dataset(self.input_data, config)
        fleet.barrier_work()
        self.example_nums = 0
        self.count_method = self.config.get("runner.example_count_method",
                                            "example")

    def train(self, dataset):
        """ training """
        train_msg = util.get_job_info()

        train_begin_time = time.time()
        for epoch in range(1, self.config.epochs + 1):
            if self.config.max_steps > 0 and model_util.print_count >= self.config.max_steps:
                log.info("training reach max_steps: %d, training end" %
                         self.config.max_steps)
                break

            epoch_begin = time.time()
            epoch_loss = 0
            pass_id = 0
            last_batch_num = 0
            for pass_dataset in dataset.pass_generator(epoch):
                train_begin = time.time()
                profiler.add_profiler_step(self.profiler_options)
                self.exe.train_from_dataset(
                    self.model_dict.train_program, pass_dataset, debug=False)
                train_end = time.time()

                batch_num = util.get_batch_num(self.model_dict.batch_count)
                log.info("train pass[%d] STAGE [TRAIN] finished, avg_batch_num[%d] time cost: %f sec" \
                        % (pass_id, batch_num - last_batch_num, train_end - train_begin))
                last_batch_num = batch_num

                t_loss = util.get_global_value(self.model_dict.visualize_loss,
                                               self.model_dict.batch_count)
                epoch_loss += t_loss
                pass_id += 1

            epoch_end = time.time()
            log.info("epoch[%d] finished, time cost: %f sec" %
                     (epoch, epoch_end - epoch_begin))

            if pass_id > 0:
                epoch_loss = epoch_loss / pass_id
            fleet.barrier_worker()
            time_msg = "%s\n" % datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            train_msg += time_msg
            msg = "Train: Epoch %s | auc: %.6f\n" % (epoch, epoch_loss)
            train_msg += msg
            log.info(msg)

            if fleet.worker_index() == 0:
                with open(os.path.join('./train_result.log'), 'a') as f:
                    f.write(time_msg)
                    f.write(msg)

            fleet.barrier_worker()

            savemodel_begin = time.time()
            is_save = (epoch % self.config.save_model_interval == 0 or
                       epoch == self.config.epochs)
            if self.config.model_save_path and is_save:
                log.info("save model for epoch {}".format(epoch))
                dataset.embedding.dump_to_mem()
                util.save_model(self.exe, self.model_dict, self.config,
                                self.config.local_model_path,
                                self.config.model_save_path)
                os.system(
                    "echo '%s' | sh ../scripts/to_robot.sh >/dev/null 2>&1 " %
                    train_msg)
            fleet.barrier_worker()
            savemodel_end = time.time()
            log.info("STAGE [SAVE MODEL] for epoch[%d] finished, time cost: %f sec" \
                % (epoch, savemodel_end - savemodel_begin))
        train_end_time = time.time()
        log.info("STAGE [TRAIN MODEL] finished, time cost: % sec" %
                 (train_end_time - train_begin_time))

    def infer(self, dataset):
        """
        infer
        """
        infer_begin = time.time()

        # set infer mode
        if hasattr(dataset.embedding.parameter_server, "set_mode"):
            dataset.embedding.set_infer_mode(True)

        for pass_dataset in dataset.pass_generator():
            self.exe.train_from_dataset(
                self.infer_model_dict.train_program, pass_dataset, debug=False)

        util.upload_embedding(self.config, self.config.local_result_path)

        infer_end = time.time()
        log.info("STAGE [INFER MODEL] finished, time cost: % sec" %
                 (infer_end - infer_begin))

    def record_result(self):
        logger.info("train_result_dict: {}".format(self.train_result_dict))
        with open("./train_result_dict.txt", 'w+') as f:
            f.write(str(self.train_result_dict))


if __name__ == "__main__":
    paddle.enable_static()
    config = parse_args()
    benchmark_main = Main(config)
    benchmark_main.run()
