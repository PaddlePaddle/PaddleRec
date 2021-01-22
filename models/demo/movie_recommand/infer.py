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

import paddle
import os
import paddle.nn as nn
import time
import logging
import sys
import importlib
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from utils.utils_single import load_yaml, load_dy_model_class, get_abs_model, create_data_loader
from utils.save_load import save_model, load_model
from paddle.io import DistributedBatchSampler, DataLoader
import argparse

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--config_yaml", type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def main(args):
    paddle.seed(12345)
    # load config
    config = load_yaml(args.config_yaml)
    dy_model_class = load_dy_model_class(args.abs_dir)
    config["config_abs_dir"] = args.abs_dir
    # tools.vars
    use_gpu = config.get("runner.use_gpu", True)
    test_data_dir = config.get("runner.test_data_dir", None)
    print_interval = config.get("runner.print_interval", None)
    model_load_path = config.get("runner.infer_load_path", "model_output")
    start_epoch = config.get("runner.infer_start_epoch", 0)
    end_epoch = config.get("runner.infer_end_epoch", 10)

    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, test_data_dir, start_epoch, end_epoch, print_interval,
               model_load_path))
    logger.info("**************common.configs**********")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    dy_model = dy_model_class.create_model(config)

    # to do : add optimizer function
    #optimizer = dy_model_class.create_optimizer(dy_model, config)

    logger.info("read data")
    test_dataloader = create_data_loader(
        config=config, place=place, mode="test")

    epoch_begin = time.time()
    interval_begin = time.time()

    metric_list, metric_list_name = dy_model_class.create_metrics()

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, dy_model)
        dy_model.eval()
        runner_results = []
        for batch_id, batch in enumerate(test_dataloader()):
            batch_size = len(batch[0])

            metric_list, tensor_print_dict, batch_runner_result = dy_model_class.infer_forward(
                dy_model, metric_list, batch, config)
            runner_results.append(batch_runner_result)

            if batch_id % print_interval == 0:
                tensor_print_str = ""
                if tensor_print_dict is not None:
                    for var_name, var in tensor_print_dict.items():
                        tensor_print_str += (
                            "{}:".format(var_name) + str(var.numpy()) + ",")
                metric_str = ""
                for metric_id in range(len(metric_list_name)):
                    metric_str += (
                        metric_list_name[metric_id] +
                        ": {:.6f},".format(metric_list[metric_id].accumulate())
                    )
                logger.info("epoch: {}, batch_id: {}, ".format(
                    epoch_id, batch_id) + metric_str + tensor_print_str +
                            " speed: {:.2f} ins/s".format(
                                print_interval * batch_size / (time.time(
                                ) - interval_begin)))
                interval_begin = time.time()

        metric_str = ""
        for metric_id in range(len(metric_list_name)):
            metric_str += (
                metric_list_name[metric_id] +
                ": {:.6f},".format(metric_list[metric_id].accumulate()))

        tensor_print_str = ""
        if tensor_print_dict is not None:
            for var_name, var in tensor_print_dict.items():
                tensor_print_str += (
                    "{}:".format(var_name) + str(var.numpy()) + ",")

        logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                    tensor_print_str + " epoch time: {:.2f} s".format(
                        time.time() - epoch_begin))

        runner_result_save_path = config.get("runner.runner_result_dump_path",
                                             None)
        if runner_result_save_path:
            logging.info("Dump runner result in {}".format(
                runner_result_save_path))
            with open(runner_result_save_path, 'w+') as fout:
                json.dump(runner_results, fout)

        epoch_begin = time.time()


if __name__ == '__main__':
    args = parse_args()
    main(args)
