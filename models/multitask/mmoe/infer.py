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
import mmoe_net as net
import time
import logging

from utils import load_yaml, get_abs_model, save_model, load_model
from census_reader_dygraph import CensusDataset
from paddle.io import DistributedBatchSampler, DataLoader
import argparse

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--config_yaml", type=str)
    args = parser.parse_args()
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def create_feeds(batch, feature_size):
    input_data = paddle.to_tensor(batch[0].numpy().astype('float32').reshape(
        -1, feature_size))
    label_income = paddle.to_tensor(batch[1].numpy().astype('float32').reshape(
        -1, 1))
    label_marital = paddle.to_tensor(batch[2].numpy().astype('float32')
                                     .reshape(-1, 1))
    return input_data, label_income, label_marital


def create_model(config):
    feature_size = config.get('hyper_parameters.feature_size', None)
    expert_num = config.get('hyper_parameters.expert_num', None)
    expert_size = config.get('hyper_parameters.expert_size', None)
    tower_size = config.get('hyper_parameters.tower_size', None)
    gate_num = config.get('hyper_parameters.gate_num', None)

    MMoE = net.MMoELayer(feature_size, expert_num, expert_size, tower_size,
                         gate_num)

    return MMoE


def create_data_loader(dataset, mode, place, config):
    batch_size = config.get('dygraph.batch_size', None)
    is_train = mode == 'train'
    batch_sampler = DistributedBatchSampler(
        dataset, batch_size=batch_size, shuffle=is_train)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, places=place)
    return loader


def main(args):
    paddle.seed(12345)
    config = load_yaml(args.config_yaml)
    use_gpu = config.get("use_gpu", True)
    test_data_dir = config.get("dygraph.test_data_dir", None)
    feature_size = config.get('hyper_parameters.feature_size', None)
    print_interval = config.get("dygraph.print_interval", None)
    model_load_path = config.get("dygraph.infer_load_path", "model_output")
    start_epoch = config.get("dygraph.infer_start_epoch", -1)
    end_epoch = config.get("dygraph.infer_end_epoch", 10)

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    mmoe_model = create_model(config)
    file_list = [
        os.path.join(test_data_dir, x) for x in os.listdir(test_data_dir)
    ]
    print("read data")
    dataset = CensusDataset(file_list)
    test_dataloader = create_data_loader(
        dataset, mode='test', place=place, config=config)

    auc_metric_marital = paddle.metric.Auc("ROC")
    auc_metric_income = paddle.metric.Auc("ROC")
    epoch_begin = time.time()
    interval_begin = time.time()

    for epoch_id in range(start_epoch + 1, end_epoch):

        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, mmoe_model)
        for batch_id, batch in enumerate(test_dataloader()):
            batch_size = len(batch[0])

            input_data, label_income, label_marital = create_feeds(
                batch, feature_size)

            pred_income, pred_marital = mmoe_model(input_data)

            # for auc
            auc_metric_income.update(
                preds=pred_income.numpy(), labels=label_income.numpy())
            auc_metric_marital.update(
                preds=pred_marital.numpy(), labels=label_marital.numpy())

            if batch_id % print_interval == 1:
                logger.info(
                    "infer epoch: {}, batch_id: {}, auc_income: {:.6f}, auc_marital: {:.6f}, speed: {:.2f} ins/s".
                    format(epoch_id, batch_id,
                           auc_metric_income.accumulate(),
                           auc_metric_marital.accumulate(), print_interval *
                           batch_size / (time.time() - interval_begin)))
                interval_begin = time.time()

        logger.info(
            "infer epoch: {} done, auc_income: {:.6f}, auc_marital: {:.6f}, : epoch time{:.2f} s".
            format(epoch_id,
                   auc_metric_income.accumulate(),
                   auc_metric_marital.accumulate(), time.time() - epoch_begin))


if __name__ == '__main__':
    args = parse_args()
    main(args)
