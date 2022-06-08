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
import time
import logging
import sys
from math import sqrt

__dir__ = os.path.dirname(os.path.abspath(__file__))
print(os.path.abspath('/'.join(__dir__.split('/')[:-3])))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath('/'.join(__dir__.split('/')[:-3])))

from tools.utils.utils_single import load_yaml, load_dy_model_class, \
    get_abs_model
from tools.utils.save_load import load_model
from paddle.io import DataLoader
import argparse
from importlib import import_module

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def create_data_loader(config, place):
    config['runner.mode'] = 'test'
    train_data_dir = config.get("runner.train_data_dir", None)
    train_file_list = [
        os.path.join(train_data_dir, x) for x in os.listdir(train_data_dir)
    ]

    test_data_dir = config.get("runner.test_data_dir", None)
    batch_size = config.get('runner.infer_batch_size', None)
    reader_path = config.get('runner.infer_reader_path', 'reader')
    test_file_list = [
        os.path.join(test_data_dir, x) for x in os.listdir(test_data_dir)
    ]
    logger.info("reader path:{}".format(reader_path))
    reader_class = import_module(reader_path)
    dataset = reader_class.RecDataset(
        train_file_list, config=config, test_list=test_file_list)
    loader = DataLoader(
        dataset, batch_size=batch_size, places=place, drop_last=True)
    return loader


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
    test_data_dir = config.get("runner.test_data_dir", None)
    print_interval = config.get("runner.print_interval", None)
    infer_batch_size = config.get("runner.infer_batch_size", 1)
    model_load_path = config.get("runner.infer_load_path", "model_output")
    start_epoch = config.get("runner.infer_start_epoch", 0)
    end_epoch = config.get("runner.infer_end_epoch", 10)

    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, use_xpu: {}, use_visual: {}, infer_batch_size: {}, \
            test_data_dir: {}, start_epoch: {}, end_epoch: {}, \
                print_interval: {}, model_load_path: {}"
        .format(use_gpu, use_xpu, use_visual, infer_batch_size, test_data_dir,
                start_epoch, end_epoch, print_interval, model_load_path))
    logger.info("**************common.configs**********")

    if use_xpu:
        xpu_device = 'xpu:{0}'.format(os.getenv('FLAGS_selected_xpus', 0))
        place = paddle.set_device(xpu_device)
    else:
        place = paddle.set_device('gpu' if use_gpu else 'cpu')

    dy_model = dy_model_class.create_model(config)

    logger.info("read data")
    test_dataloader = create_data_loader(config=config, place=place)

    epoch_begin = time.time()
    interval_begin = time.time()

    metric_list, metric_list_name = dy_model_class.create_metrics()
    step_num = 0

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, dy_model)
        dy_model.eval()
        infer_reader_cost = 0.0
        infer_run_cost = 0.0
        reader_start = time.time()

        denom = 0.0
        n = 0
        for batch_id, batch in enumerate(test_dataloader()):
            infer_reader_cost += time.time() - reader_start
            infer_start = time.time()

            metric_list, tensor_print_dict = dy_model_class.infer_forward(
                dy_model, metric_list, batch, config)

            infer_run_cost += time.time() - infer_start

            if batch_id % print_interval == 0:
                logger.info(
                    "epoch: {}, batch_id: {}, ".format(epoch_id, batch_id) +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, \
                    avg_samples: {:.5f}, ips: {:.2f} ins/s"
                    .format(infer_reader_cost / print_interval, (
                        infer_reader_cost + infer_run_cost) / print_interval,
                            infer_batch_size, print_interval * infer_batch_size
                            / (time.time() - interval_begin)))
                interval_begin = time.time()
                infer_reader_cost = 0.0
                infer_run_cost = 0.0
            step_num = step_num + 1

            denom += tensor_print_dict['SE']
            n += tensor_print_dict['num']

        metric_str = "RMSE: %.5f" % sqrt(denom / n)

        tensor_print_str = ""

        logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                    tensor_print_str + " epoch time: {:.2f} s".format(
                        time.time() - epoch_begin))
        epoch_begin = time.time()


if __name__ == '__main__':
    args = parse_args()
    main(args)
