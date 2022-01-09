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
print(os.path.abspath('/'.join(__dir__.split('/')[:-3])))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath('/'.join(__dir__.split('/')[:-3])))

from tools.utils.utils_single import load_yaml, load_dy_model_class, get_abs_model
from tools.utils.save_load import save_model, load_model
from paddle.io import DistributedBatchSampler, DataLoader
import argparse
import numpy as np
from importlib import import_module

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# def create_data_loader(args):
#     data_dir = args['runner.train_data_dir']
#     reader_path, reader_file = os.path.split(args.reader_file)
#     reader_file, extension = os.path.splitext(reader_file)
#     batchsize = args['runner.infer_batch_size']
#     file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
#     sys.path.append(reader_path)
#     reader_class = import_module(reader_file)
#     config = {"runner.inference": True, }
#     dataset = reader_class.RecDataset(file_list, config=config)
#     loader = DataLoader(
#         dataset, batch_size=batchsize,drop_last=False)
#     return loader


def create_data_loader(config, place, mode="train"):
    config['runner.mode'] = 'test'
    data_dir = config.get("runner.test_data_dir", None)
    batch_size = config.get('runner.infer_batch_size', None)
    reader_path = config.get('runner.infer_reader_path', 'reader')
    config_abs_dir = config.get("config_abs_dir", None)
    data_dir = os.path.join(config_abs_dir, data_dir)
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    user_define_reader = config.get('runner.user_define_reader', False)
    logger.info("reader path:{}".format(reader_path))
    from importlib import import_module
    reader_class = import_module(reader_path)
    dataset = reader_class.RecDataset(file_list, config=config)
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
    infer_batch_size = config.get("runner.infer_batch_size", None)
    model_load_path = config.get("runner.infer_load_path", "model_output")
    start_epoch = config.get("runner.infer_start_epoch", 0)
    end_epoch = config.get("runner.infer_end_epoch", 10)

    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, use_xpu: {}, use_visual: {}, infer_batch_size: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, use_xpu, use_visual, infer_batch_size, test_data_dir,
               start_epoch, end_epoch, print_interval, model_load_path))
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
        log_visual = LogWriter(args.abs_dir + "/visualDL_log/infer")

    # to do : add optimizer function
    # optimizer = dy_model_class.create_optimizer(dy_model, config)

    logger.info("read data")
    print(config)
    test_dataloader = create_data_loader(config, place)

    epoch_begin = time.time()
    interval_begin = time.time()

    metric_list, metric_list_name = dy_model_class.create_metrics()
    step_num = 0
    dataset = test_dataloader.dataset

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, dy_model)
        dy_model.eval()
        infer_reader_cost = 0.0
        infer_run_cost = 0.0
        reader_start = time.time()
        NDCG = 0.0
        HT = 0.0
        valid_user = 0.0
        pred = []
        for batch_id, batch in enumerate(test_dataloader()):
            valid_user += 1
            infer_reader_cost += time.time() - reader_start
            infer_start = time.time()
            metric_list, tensor_print_dict = dy_model_class.infer_forward(
                dy_model, metric_list, batch, config)

            infer_run_cost += time.time() - infer_start

            if batch_id % print_interval == 0:
                logger.info(
                    "epoch: {}, batch_id: {}, ".format(epoch_id, batch_id) +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.2f} ins/s".
                    format(infer_reader_cost / print_interval, (
                        infer_reader_cost + infer_run_cost) / print_interval,
                           infer_batch_size, print_interval * infer_batch_size
                           / (time.time() - interval_begin)))
                interval_begin = time.time()
                infer_reader_cost = 0.0
                infer_run_cost = 0.0
            step_num = step_num + 1
            # pred.append(-tensor_print_dict['prediction'])
            predictions = -tensor_print_dict['prediction'][0]
            rank = predictions.argsort().argsort()[0].item()
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
        NDCG = NDCG / valid_user
        HT = HT / valid_user
        # pred = paddle.concat(pred, 0)
        # logger.info(pred.shape)
        # rank = pred.argsort().argsort()[:, 0].cpu().numpy()
        # HT = (rank < 10).mean()
        # NDCG = (1 / np.log(rank[rank < 10] + 2) ).sum()/ len(rank)
        # logger.info(rank.shape)
        metric_str = "NDCG@10 : {:.4f},".format(
            NDCG) + " HR@10 : {:.4f},".format(HT)

        tensor_print_str = ""

        logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                    tensor_print_str + " epoch time: {:.2f} s".format(
                        time.time() - epoch_begin))
        epoch_begin = time.time()


if __name__ == '__main__':
    args = parse_args()
    main(args)
