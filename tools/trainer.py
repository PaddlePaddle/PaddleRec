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

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from tools.utils.utils import load_yaml, save_model, load_dy_program, get_abs_model
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


def create_data_loader(config, place):
    train_data_dir = config.get("dygraph.train_data_dir", None)
    config_abs_dir = config.get("config_abs_dir", None)
    train_data_dir = os.path.join(config_abs_dir, train_data_dir)
    file_list = [
        os.path.join(train_data_dir, x) for x in os.listdir(train_data_dir)
    ]
    datasets_name = config.get('dygraph.dataset')
    datasets = importlib.import_module(datasets_name)
    dataset = datasets.DygraphDataset(file_list)
    batch_size = config.get('dygraph.batch_size', None)
    loader = DataLoader(
        dataset, batch_size=batch_size, places=place, drop_last=True)
    return loader


def main(args):
    paddle.seed(12345)
    # load config
    config = load_yaml(args.config_yaml)
    dy_program_class = load_dy_program(args.abs_dir)
    config["config_abs_dir"] = args.abs_dir
    # tools.vars
    use_gpu = config.get("dygraph.use_gpu", True)
    train_data_dir = config.get("dygraph.train_data_dir", None)
    epochs = config.get("dygraph.epochs", None)
    print_interval = config.get("dygraph.print_interval", None)
    model_save_path = config.get("dygraph.model_save_path", "model_output")
    model_init_path = config.get("dygraph.model_init_path", None)
    #dense_input_dim = config.get('hyper_parameters.dense_input_dim', None)

    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, train_data_dir, epochs, print_interval,
               model_save_path))
    logger.info("**************common.configs**********")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    dy_model = dy_program_class.create_model(config)

    if model_init_path is not None:
        load_model(model_init_path, dy_model)

    # to do : add optimizer function
    optimizer = dy_program_class.create_optimizer(dy_model, config)

    logger.info("read data")
    train_dataloader = create_data_loader(config=config, place=place)

    last_epoch_id = config.get("last_epoch", -1)

    for epoch_id in range(last_epoch_id + 1, epochs):
        # set train mode
        dy_model.train()
        metric_list, metric_list_name = dy_program_class.create_metrics()
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

            loss, metric_list = dy_program_class.train_forward(
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
                        ": {:.6f},".format(metric_list[metric_id].accumulate())
                    )
                logger.info(
                    "epoch: {}, batch_id: {}, ".format(epoch_id,
                                                       batch_id) + metric_str +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(train_reader_cost / print_interval, (
                        train_reader_cost + train_run_cost) / print_interval,
                           total_samples / print_interval, total_samples / (
                               train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()

        metric_str = ""
        for metric_id in range(len(metric_list_name)):
            metric_str += (
                metric_list_name[metric_id] +
                ": {:.6f},".format(metric_list[metric_id].accumulate()))

        logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                    " : epoch time{:.2f} s".format(time.time() - epoch_begin))

        save_model(
            dy_model, optimizer, model_save_path, epoch_id, prefix='rec')


if __name__ == '__main__':
    args = parse_args()
    main(args)
