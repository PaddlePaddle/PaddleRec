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
import dssm_net as net
import time
import logging
from utils import load_yaml, get_abs_model, save_model, load_model
from synthetic_reader_dygraph import SyntheticDataset
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


def create_feeds(batch, trigram_d):
    query = paddle.to_tensor(batch[0].numpy().astype('float32').reshape(
        -1, trigram_d))
    doc_pos = paddle.to_tensor(batch[1].numpy().astype('float32').reshape(
        -1, trigram_d))
    doc_negs = []
    for ele in batch[2:]:
        doc_negs.append(
            paddle.to_tensor(ele.numpy().astype('float32').reshape(-1,
                                                                   trigram_d)))
    return query, doc_pos, doc_negs


def create_loss(hit_prob):
    loss = -paddle.sum(paddle.log(hit_prob))
    avg_cost = paddle.mean(x=loss)
    return avg_cost


def create_model(config):
    trigram_d = config.get('hyper_parameters.trigram_d', None)
    neg_num = config.get('hyper_parameters.neg_num', None)
    slice_end = config.get('hyper_parameters.slice_end', None)
    fc_sizes = config.get('hyper_parameters.fc_sizes', None)
    fc_acts = config.get('hyper_parameters.fc_acts', None)

    DSSM = net.DSSMLayer(trigram_d, neg_num, slice_end, fc_sizes, fc_acts)

    return DSSM


def create_data_loader(dataset, place, config):
    batch_size = config.get('dygraph.batch_size', None)
    loader = DataLoader(dataset, batch_size=batch_size, places=place)
    return loader


def main(args):
    paddle.seed(12345)
    config = load_yaml(args.config_yaml)
    use_gpu = config.get("dygraph.use_gpu", False)
    train_data_dir = config.get("dygraph.train_data_dir", None)
    epochs = config.get("dygraph.epochs", None)
    print_interval = config.get("dygraph.print_interval", None)
    model_save_path = config.get("dygraph.model_save_path", "model_output")
    trigram_d = config.get('hyper_parameters.trigram_d', None)
    batch_size = config.get('dygraph.batch_size', None)

    print("***********************************")
    logger.info(
        "use_gpu: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, train_data_dir, epochs, print_interval,
               model_save_path))
    print("***********************************")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    dssm_model = create_model(config)
    model_init_path = config.get("dygraph.model_init_path", None)
    if model_init_path is not None:
        load_model(model_init_path, dssm_model)

    # to do : add optimizer function
    learning_rate = config.get("hyper_parameters.optimizer.learning_rate",
                               0.001)
    optimizer = paddle.optimizer.Adam(
        learning_rate=learning_rate, parameters=dssm_model.parameters())

    # to do init model
    file_list = [
        os.path.join(train_data_dir, x) for x in os.listdir(train_data_dir)
    ]
    print("read data")
    dataset = SyntheticDataset(file_list)
    train_dataloader = create_data_loader(dataset, place=place, config=config)

    last_epoch_id = config.get("last_epoch", -1)

    for epoch_id in range(last_epoch_id + 1, epochs):
        # set train mode
        dssm_model.train()
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

            query, doc_pos, doc_negs = create_feeds(batch, trigram_d)

            inputs = [query, doc_pos] + doc_negs
            R_Q_D_p, hit_prob = dssm_model(inputs, False)
            loss = create_loss(hit_prob)

            loss.backward()
            optimizer.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size

            if batch_id % print_interval == 0:
                logger.info(
                    "epoch: {}, batch_id: {}, loss: {}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(epoch_id, batch_id,
                           loss.numpy(), train_reader_cost / print_interval, (
                               train_reader_cost + train_run_cost) /
                           print_interval, total_samples / print_interval,
                           total_samples / (train_reader_cost + train_run_cost
                                            )))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()

        logger.info("epoch: {} done, loss: {}, : epoch time{:.2f} s".format(
            epoch_id, loss.numpy(), time.time() - epoch_begin))

        save_model(
            dssm_model, optimizer, model_save_path, epoch_id, prefix='rec')


if __name__ == '__main__':
    args = parse_args()
    main(args)
