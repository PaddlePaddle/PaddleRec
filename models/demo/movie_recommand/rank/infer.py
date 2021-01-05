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
import rank_net as net
import time
import json
import logging
import paddle.nn.functional as F
from utils import load_yaml, get_abs_model, save_model, load_model
from movie_reader_dygraph import MovieDataset
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


def create_feeds(batch):
    user_sparse_inputs = [
        paddle.to_tensor(batch[i].numpy().astype('int64').reshape(-1, 1))
        for i in range(4)
    ]

    mov_sparse_inputs = [
        paddle.to_tensor(batch[4].numpy().astype('int64').reshape(-1, 1)),
        paddle.to_tensor(batch[5].numpy().astype('int64').reshape(-1, 4)),
        paddle.to_tensor(batch[6].numpy().astype('int64').reshape(-1, 3))
    ]

    label_input = paddle.to_tensor(batch[7].numpy().astype('int64').reshape(-1,
                                                                            1))

    return user_sparse_inputs, mov_sparse_inputs, label_input


def create_model(config):
    sparse_feature_number = config.get(
        "hyper_parameters.sparse_feature_number")
    sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
    fc_sizes = config.get("hyper_parameters.fc_sizes")

    rank_model = net.DNNLayer(sparse_feature_number, sparse_feature_dim,
                              fc_sizes)

    return rank_model


def create_data_loader(dataset, place, config):
    batch_size = config.get('dygraph.batch_size', None)
    loader = DataLoader(dataset, batch_size=batch_size, places=place)
    return loader


def main(args):
    paddle.seed(12345)
    config = load_yaml(args.config_yaml)
    use_gpu = config.get("dygraph.use_gpu", False)
    test_data_dir = config.get("dygraph.test_data_dir", None)
    print_interval = config.get("dygraph.print_interval", None)
    model_load_path = config.get("dygraph.infer_load_path", "increment_rank")
    start_epoch = config.get("dygraph.infer_start_epoch", 3)
    end_epoch = config.get("dygraph.infer_end_epoch", 5)
    batch_size = config.get("dygraph.batch_size", 128)

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    print("***********************************")
    logger.info(
        "use_gpu: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, test_data_dir, start_epoch, end_epoch, print_interval,
               model_load_path))
    print("***********************************")

    rank_model = create_model(config)
    file_list = [
        os.path.join(test_data_dir, x) for x in os.listdir(test_data_dir)
    ]
    print("read data")
    dataset = MovieDataset(file_list)
    test_dataloader = create_data_loader(dataset, place=place, config=config)

    epoch_begin = time.time()
    interval_begin = time.time()

    for epoch_id in range(start_epoch + 1, end_epoch):

        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, rank_model)
        runner_results = []
        for batch_id, batch in enumerate(test_dataloader()):
            batch_size = config.get("dygraph.batch_size", 128)
            batch_runner_result = {}

            user_sparse_inputs, mov_sparse_inputs, label_input = create_feeds(
                batch)

            predict = rank_model(batch_size, user_sparse_inputs,
                                 mov_sparse_inputs, label_input)

            uid = user_sparse_inputs[0]
            movieid = mov_sparse_inputs[0]
            label = label_input
            predict = predict

            if batch_id % print_interval == 0:
                logger.info(
                    "infer epoch: {}, batch_id: {}, uid: {}, movieid: {}, label: {}, predict: {},speed: {:.2f} ins/s".
                    format(epoch_id, batch_id,
                           uid.numpy(),
                           movieid.numpy(),
                           label.numpy(),
                           predict.numpy(), print_interval * batch_size / (
                               time.time() - interval_begin)))
                interval_begin = time.time()

            batch_runner_result["userid"] = uid.numpy().tolist()
            batch_runner_result["movieid"] = movieid.numpy().tolist()
            batch_runner_result["label"] = label.numpy().tolist()
            batch_runner_result["predict"] = predict.numpy().tolist()
            runner_results.append(batch_runner_result)

        logger.info("infer epoch: {} done, epoch time: {:.2f} s".format(
            epoch_id, time.time() - epoch_begin))

        runner_result_save_path = config.get("dygraph.runner_result_dump_path",
                                             None)
        if runner_result_save_path:
            logging.info("Dump runner result in {}".format(
                runner_result_save_path))
            with open(runner_result_save_path, 'w+') as fout:
                json.dump(runner_results, fout)


if __name__ == '__main__':
    args = parse_args()
    main(args)
