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
import simnet_net as net
import time
import logging
from utils import load_yaml, get_abs_model, save_model, load_model
from reader_dygraph import BQDataset
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


def create_feeds(batch, query_len, pos_len, neg_len):
    q_slots = [
        paddle.to_tensor(batch[0].numpy().astype('int64').reshape(-1,
                                                                  query_len))
    ]
    pt_slots = [
        paddle.to_tensor(batch[1].numpy().astype('int64').reshape(-1, pos_len))
    ]
    nt_slots = [
        paddle.to_tensor(batch[2].numpy().astype('int64').reshape(-1, neg_len))
    ]
    inputs = [q_slots, pt_slots, nt_slots]
    return inputs


def create_loss(batch_size, margin, cos_pos, cos_neg):
    loss_part1 = paddle.subtract(
        paddle.full(
            shape=[batch_size, 1], fill_value=margin, dtype='float32'),
        cos_pos)

    loss_part2 = paddle.add(loss_part1, cos_neg)

    loss_part3 = paddle.maximum(
        paddle.full(
            shape=[batch_size, 1], fill_value=0.0, dtype='float32'),
        loss_part2)

    avg_cost = paddle.mean(loss_part3)
    return avg_cost


def create_model(config):
    query_encoder = config.get('hyper_parameters.query_encoder', "gru")
    title_encoder = config.get('hyper_parameters.title_encoder', "gru")
    query_encode_dim = config.get('hyper_parameters.query_encode_dim', 128)
    title_encode_dim = config.get('hyper_parameters.title_encode_dim', 128)
    emb_size = config.get('hyper_parameters.sparse_feature_dim', 6327)
    emb_dim = config.get('hyper_parameters.emb_dim', 128)
    hidden_size = config.get('hyper_parameters.hidden_size', 128)
    margin = config.get('hyper_parameters.margin', 0.1)
    query_len = config.get('hyper_parameters.query_len', 79)
    pos_len = config.get('hyper_parameters.pos_len', 99)
    neg_len = config.get('hyper_parameters.neg_len', 90)

    simnet_model = net.MultiviewSimnetLayer(
        query_encoder, title_encoder, query_encode_dim, title_encode_dim,
        emb_size, emb_dim, hidden_size, margin, query_len, pos_len, neg_len)

    return simnet_model


def create_data_loader(dataset, place, config):
    batch_size = config.get('dygraph.batch_size_train', None)
    loader = DataLoader(dataset, batch_size=batch_size, places=place)
    return loader


def get_acc(x, y, batch_size):
    less = paddle.cast(paddle.less_than(x, y), dtype='float32')
    label_ones = paddle.full(
        dtype='float32', shape=[batch_size, 1], fill_value=1.0)
    correct = paddle.sum(less)
    total = paddle.sum(label_ones)
    acc = paddle.divide(correct, total)
    return acc


def main(args):
    paddle.seed(12345)
    config = load_yaml(args.config_yaml)
    use_gpu = config.get("dygraph.use_gpu", False)
    train_data_dir = config.get("dygraph.train_data_dir", None)
    epochs = config.get("dygraph.epochs", None)
    print_interval = config.get("dygraph.print_interval", None)
    model_save_path = config.get("dygraph.model_save_path", "model_output")
    batch_size = config.get('dygraph.batch_size_train', None)
    margin = config.get('hyper_parameters.margin', 0.1)
    query_len = config.get('hyper_parameters.query_len', 79)
    pos_len = config.get('hyper_parameters.pos_len', 99)
    neg_len = config.get('hyper_parameters.neg_len', 90)

    print("***********************************")
    logger.info(
        "use_gpu: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, train_data_dir, epochs, print_interval,
               model_save_path))
    print("***********************************")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    simnet_model = create_model(config)
    model_init_path = config.get("dygraph.model_init_path", None)
    if model_init_path is not None:
        load_model(model_init_path, simnet_model)

    # to do : add optimizer function
    learning_rate = config.get("hyper_parameters.optimizer.learning_rate",
                               0.001)
    optimizer = paddle.optimizer.Adam(
        learning_rate=learning_rate, parameters=simnet_model.parameters())

    # to do init model
    file_list = [
        os.path.join(train_data_dir, x) for x in os.listdir(train_data_dir)
    ]
    print("read data")
    dataset = BQDataset(file_list)
    train_dataloader = create_data_loader(dataset, place=place, config=config)

    last_epoch_id = config.get("last_epoch", -1)

    for epoch_id in range(last_epoch_id + 1, epochs):
        # set train mode
        simnet_model.train()
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
            batch_size = batch_size

            inputs = create_feeds(batch, query_len, pos_len, neg_len)

            cos_pos, cos_neg = simnet_model(inputs, False)
            loss = create_loss(batch_size, margin, cos_pos, cos_neg)
            acc = get_acc(cos_neg, cos_pos, batch_size)

            loss.backward()
            optimizer.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size

            if batch_id % print_interval == 0:
                logger.info(
                    "epoch: {}, batch_id: {}, acc: {}, loss: {}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(epoch_id, batch_id,
                           acc.numpy(),
                           loss.numpy(), train_reader_cost / print_interval, (
                               train_reader_cost + train_run_cost) /
                           print_interval, total_samples / print_interval,
                           total_samples / (train_reader_cost + train_run_cost
                                            )))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()

        logger.info(
            "epoch: {} done, acc: {}, loss: {}, : epoch time{:.2f} s".format(
                epoch_id, acc.numpy(), loss.numpy(), time.time(
                ) - epoch_begin))

        save_model(
            simnet_model, optimizer, model_save_path, epoch_id, prefix='rec')


if __name__ == '__main__':
    args = parse_args()
    main(args)
