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

import paddle
import os
import paddle.nn as nn
import time
import logging
import sys
import importlib

__dir__ = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from utils.utils_single import load_yaml, load_dy_model_class, get_abs_model, create_data_loader
from utils.save_load import load_model, save_model
from paddle.io import DistributedBatchSampler, DataLoader
import argparse

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--config_yaml", type=str)
    parser.add_argument("-o", "--opt", nargs='*', type=str)
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

    # tools.vars
    use_gpu = config.get("runner.use_gpu", True)
    use_xpu = config.get("runner.use_xpu", False)
    use_visual = config.get("runner.use_visual", False)
    train_data_dir = config.get("runner.train_data_dir", None)
    epochs = config.get("runner.epochs", None)
    print_interval = config.get("runner.print_interval", None)
    train_batch_size = config.get("runner.train_batch_size", None)
    model_save_path = config.get("runner.model_save_path", "model_output")
    model_init_path = config.get("runner.model_init_path", None)
    use_fleet = config.get("runner.use_fleet", False)

    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, use_xpu: {}, use_visual: {}, train_batch_size: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, use_xpu, use_visual, train_batch_size, train_data_dir,
               epochs, print_interval, model_save_path))
    logger.info("**************common.configs**********")

    if use_xpu:
        xpu_device = 'xpu:{0}'.format(os.getenv('FLAGS_selected_xpus', 0))
        place = paddle.set_device(xpu_device)
    else:
        place = paddle.set_device('gpu' if use_gpu else 'cpu')

    dy_model = dy_model_class.create_model(config)

    # Create a log_visual object and store the data in the path
    if use_visual:
        from visualdl import LogWriter
        log_visual = LogWriter(args.abs_dir + "/visualDL_log/train")

    if model_init_path is not None:
        load_model(model_init_path, dy_model)

    # to do : add optimizer function
    optimizer = dy_model_class.create_optimizer(dy_model, config)

    # use fleet run collective
    if use_fleet:
        from paddle.distributed import fleet
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)
        optimizer = fleet.distributed_optimizer(optimizer)
        dy_model = fleet.distributed_model(dy_model)

    logger.info("read data")
    train_dataloader = create_data_loader(config=config, place=place)

    last_epoch_id = config.get("last_epoch", -1)
    step_num = 0

    for epoch_id in range(last_epoch_id + 1, epochs):
        # set train mode
        dy_model.train()
        metric_list, metric_list_name = dy_model_class.create_metrics()
        #auc_metric = paddle.metric.Auc("ROC")
        epoch_begin = time.time()
        interval_begin = time.time()
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()

        for batch_id, batch in enumerate(train_dataloader()):
            train_reader_cost += time.time() - reader_start
            optimizer.clear_grad()
            train_start = time.time()
            batch_size = len(batch[0])

            loss, metric_list, tensor_print_dict = dy_model_class.train_forward(
                dy_model, metric_list, batch, config)

            loss.backward()
            optimizer.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size

            if batch_id % print_interval == 0:
                metric_str = ""
                for metric_id in range(len(metric_list_name)):
                    metric_str += (
                        metric_list_name[metric_id] +
                        ":{:.6f}, ".format(metric_list[metric_id].accumulate())
                    )
                    if use_visual:
                        log_visual.add_scalar(
                            tag="train/" + metric_list_name[metric_id],
                            step=step_num,
                            value=metric_list[metric_id].accumulate())
                tensor_print_str = ""
                if tensor_print_dict is not None:
                    for var_name, var in tensor_print_dict.items():
                        tensor_print_str += (
                            "{}:".format(var_name) +
                            str(var.numpy()).strip("[]") + ",")
                        if use_visual:
                            log_visual.add_scalar(
                                tag="train/" + var_name,
                                step=step_num,
                                value=var.numpy())
                logger.info(
                    "epoch: {}, batch_id: {}, ".format(
                        epoch_id, batch_id) + metric_str + tensor_print_str +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} ins/s".
                    format(train_reader_cost / print_interval, (
                        train_reader_cost + train_run_cost) / print_interval,
                           total_samples / print_interval, total_samples / (
                               train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()
            step_num = step_num + 1

        metric_str = ""
        for metric_id in range(len(metric_list_name)):
            metric_str += (
                metric_list_name[metric_id] +
                ": {:.6f},".format(metric_list[metric_id].accumulate()))

        tensor_print_str = ""
        if tensor_print_dict is not None:
            for var_name, var in tensor_print_dict.items():
                tensor_print_str += (
                    "{}:".format(var_name) + str(var.numpy()).strip("[]") + ","
                )

        logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                    tensor_print_str + " epoch time: {:.2f} s".format(
                        time.time() - epoch_begin))

        if use_fleet:
            trainer_id = paddle.distributed.get_rank()
            if trainer_id == 0:
                save_model(
                    dy_model,
                    optimizer,
                    model_save_path,
                    epoch_id,
                    prefix='rec')
        else:
            save_model(
                dy_model, optimizer, model_save_path, epoch_id, prefix='rec')


if __name__ == '__main__':
    args = parse_args()
    main(args)
