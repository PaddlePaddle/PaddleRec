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
import warnings
import logging
import paddle
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from utils.static_ps.reader_helper import get_reader
from utils.utils_single import load_yaml, load_static_model_class, get_abs_model, create_data_loader, reset_auc
from utils.save_load import save_static_model

import time
import argparse
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("PaddleRec train static script")
    parser.add_argument("-m", "--config_yaml", type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def main(args):
    paddle.seed(12345)

    # load config
    config = load_yaml(args.config_yaml)
    config["yaml_path"] = args.config_yaml
    config["config_abs_dir"] = args.abs_dir
    # load static model class
    static_model_class = load_static_model_class(config)

    input_data = static_model_class.create_feeds()
    input_data_names = [data.name for data in input_data]

    fetch_vars = static_model_class.net(input_data)
    #infer_target_var = model.infer_target_var
    logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))
    static_model_class.create_optimizer()

    use_gpu = config.get("runner.use_gpu", True)
    use_auc = config.get("runner.use_auc", False)
    train_data_dir = config.get("runner.train_data_dir", None)
    epochs = config.get("runner.epochs", None)
    print_interval = config.get("runner.print_interval", None)
    model_save_path = config.get("runner.model_save_path", "model_output")
    model_init_path = config.get("runner.model_init_path", None)
    batch_size = config.get("runner.train_batch_size", None)
    reader_type = config.get("runner.reader_type", "DataLoader")
    os.environ["CPU_NUM"] = str(config.get("runner.thread_num", 1))
    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, train_data_dir, epochs, print_interval,
               model_save_path))
    logger.info("**************common.configs**********")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')
    exe = paddle.static.Executor(place)
    # initialize
    exe.run(paddle.static.default_startup_program())

    last_epoch_id = config.get("last_epoch", -1)

    if reader_type == 'QueueDataset':
        dataset, file_list = get_reader(input_data, config)
    elif reader_type == 'DataLoader':
        train_dataloader = create_data_loader(config=config, place=place)

    for epoch_id in range(last_epoch_id + 1, epochs):

        epoch_begin = time.time()
        if use_auc:
            reset_auc()

        if reader_type == 'DataLoader':
            fetch_batch_var = dataloader_train(epoch_id, train_dataloader,
                                               input_data_names, fetch_vars,
                                               exe, config)
            metric_str = ""
            for var_idx, var_name in enumerate(fetch_vars):
                metric_str += "{}: {}, ".format(var_name,
                                                fetch_batch_var[var_idx])
            logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                        "epoch time: {:.2f} s".format(time.time() -
                                                      epoch_begin))
        elif reader_type == 'QueueDataset':
            fetch_batch_var = dataset_train(epoch_id, dataset, fetch_vars, exe,
                                            config)
            logger.info("epoch: {} done, ".format(epoch_id) +
                        "epoch time: {:.2f} s".format(time.time() -
                                                      epoch_begin))
        else:
            logger.info("reader type wrong")

        save_static_model(
            paddle.static.default_main_program(),
            model_save_path,
            epoch_id,
            prefix='rec_static')


def dataset_train(epoch_id, dataset, fetch_vars, exe, config):
    #logger.info("Epoch: {}, Running Dataset Begin.".format(epoch))
    fetch_info = [
        "Epoch {} Var {}".format(epoch_id, var_name) for var_name in fetch_vars
    ]
    fetch_vars = [var for _, var in fetch_vars.items()]
    print_interval = config.get("runner.print_interval")
    exe.train_from_dataset(
        program=paddle.static.default_main_program(),
        dataset=dataset,
        fetch_list=fetch_vars,
        fetch_info=fetch_info,
        print_period=print_interval,
        debug=config.get("runner.dataset_debug"))


def dataloader_train(epoch_id, train_dataloader, input_data_names, fetch_vars,
                     exe, config):
    print_interval = config.get("runner.print_interval", None)
    batch_size = config.get("runner.train_batch_size", None)
    interval_begin = time.time()
    train_reader_cost = 0.0
    train_run_cost = 0.0
    total_samples = 0
    reader_start = time.time()
    for batch_id, batch_data in enumerate(train_dataloader()):
        train_reader_cost += time.time() - reader_start
        train_start = time.time()

        fetch_batch_var = exe.run(
            program=paddle.static.default_main_program(),
            feed=dict(zip(input_data_names, batch_data)),
            fetch_list=[var for _, var in fetch_vars.items()])
        train_run_cost += time.time() - train_start
        total_samples += batch_size
        if batch_id % print_interval == 0:
            metric_str = ""
            for var_idx, var_name in enumerate(fetch_vars):
                metric_str += "{}: {}, ".format(var_name,
                                                fetch_batch_var[var_idx])
            logger.info(
                "epoch: {}, batch_id: {}, ".format(epoch_id,
                                                   batch_id) + metric_str +
                "avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                format(train_reader_cost / print_interval, (
                    train_reader_cost + train_run_cost) / print_interval,
                       total_samples / print_interval, total_samples / (
                           train_reader_cost + train_run_cost)))
            train_reader_cost = 0.0
            train_run_cost = 0.0
            total_samples = 0
        reader_start = time.time()
    return fetch_batch_var


if __name__ == "__main__":
    paddle.enable_static()
    args = parse_args()
    main(args)
