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
import numpy as np
import warnings
import logging
import paddle
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet
import time
import argparse
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from common import YamlHelper, is_number
from program_helper import get_model, get_strategy
from reader_helper import get_reader, get_infer_reader, get_example_num, get_file_list, get_word_num

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
        self.epoch_model_path_list = []
        self.epoch_model_name_list = []

    def run(self):
        self.network()
        self.init_reader()
        use_cuda = int(config.get("runner.use_gpu"))
        place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
        self.exe = paddle.static.Executor(place)

        init_model_path = config.get("runner.model_save_path")
        for file in os.listdir(init_model_path):
            file_path = os.path.join(init_model_path, file)
            # hard code for epoch model folder
            if os.path.isdir(file_path) and is_number(file):
                self.epoch_model_path_list.append(file_path)
                self.epoch_model_name_list.append(file)

        if len(self.epoch_model_path_list) == 0:
            self.epoch_model_path_list.append(init_model_path)
            self.epoch_model_name_list.append(init_model_path)

        self.epoch_model_path_list.sort()
        self.epoch_model_name_list.sort()

        for idx, model_path in enumerate(self.epoch_model_path_list):
            logger.info("Begin Infer Model {}".format(
                self.epoch_model_name_list[idx]))
            self.run_infer(model_path, self.epoch_model_name_list[idx])
        logger.info("Run Success, Exit.")

    def network(self):
        model = get_model(self.config)
        self.input_data = model.create_feeds()
        self.init_reader()

    def run_infer(self, model_path, model_name):
        [inference_program, feed_target_names, fetch_targets] = (
            paddle.fluid.io.load_inference_model(
                dirname=model_path, executor=self.exe))

        self.reset_auc()

        for batch_id, data in enumerate(self.reader()):
            results = self.exe.run(inference_program,
                                   feed=data,
                                   fetch_list=fetch_targets)
            batch_id += 1
            print_step = int(config.get("runner.print_interval"))
            if batch_id % print_step == 0:
                metrics_string = ""
                for var_idx, var_name in enumerate(results):
                    metrics_string += "Infer res: {}, ".format(results[
                        var_idx])
                logger.info("Model: {}, Batch: {}, {}".format(
                    model_name, batch_id, metrics_string))

    def init_reader(self):
        self.reader, self.file_list = get_infer_reader(self.input_data, config)
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

    def reset_auc(self):
        auc_var_name = [
            "_generated_var_0", "_generated_var_1", "_generated_var_2",
            "_generated_var_3"
        ]
        for name in auc_var_name:
            param = paddle.static.global_scope().var(name)
            if param == None:
                continue
            tensor = param.get_tensor()
            if param:
                tensor_array = np.zeros(tensor._get_dims()).astype("int64")
                tensor.set(tensor_array, paddle.CPUPlace())
                logger.info("AUC Reset To Zero: {}".format(name))


if __name__ == "__main__":
    paddle.enable_static()
    config = parse_args()
    save_model_path = config.get("runner.model_save_path")
    if save_model_path and os.path.exists(save_model_path):
        benchmark_main = Main(config)
        benchmark_main.run()
