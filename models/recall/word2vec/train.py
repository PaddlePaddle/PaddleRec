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
import word2vec_net as net
import time
import logging

from utils_dygraph import load_yaml, get_abs_model, save_model, load_model
from word2vec_reader_dygraph import Word2VecDataset
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


def create_feeds(batch, neg_num):
    input_word = paddle.to_tensor(batch[0].numpy().astype('int64').reshape(-1,
                                                                           1))
    true_word = paddle.to_tensor(batch[1].numpy().astype('int64').reshape(-1,
                                                                          1))
    neg_word = paddle.to_tensor(batch[2].numpy().astype('int64').reshape(
        -1, neg_num))
    return input_word, true_word, neg_word


def create_loss(true_logits, neg_logits, neg_num):
    label_ones = paddle.full(
        shape=[paddle.shape(true_logits)[0], 1], fill_value=1.0)
    label_zeros = paddle.full(
        shape=[paddle.shape(true_logits)[0], neg_num], fill_value=0.0)

    true_logits = paddle.nn.functional.sigmoid(true_logits)
    true_xent = paddle.nn.functional.binary_cross_entropy(true_logits,
                                                          label_ones)
    neg_logits = paddle.nn.functional.sigmoid(neg_logits)
    neg_xent = paddle.nn.functional.binary_cross_entropy(neg_logits,
                                                         label_zeros)
    cost = paddle.add(true_xent, neg_xent)
    avg_cost = paddle.mean(x=cost)

    return avg_cost


def create_model(config):
    sparse_feature_number = config.get(
        "hyper_parameters.sparse_feature_number")
    sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
    neg_num = config.get("hyper_parameters.neg_num")

    word2vec_model = net.Word2VecLayer(
        sparse_feature_number,
        sparse_feature_dim,
        neg_num,
        emb_name="emb",
        emb_w_name="emb_w",
        emb_b_name="emb_b")

    return word2vec_model


def create_data_loader(dataset, place, config):
    batch_size = config.get('dygraph.batch_size', None)
    loader = DataLoader(
        dataset, batch_size=batch_size, places=place, drop_last=True)
    return loader


def main(args):
    paddle.seed(12345)
    config = load_yaml(args.config_yaml)
    use_gpu = config.get("dygraph.use_gpu", True)
    train_data_dir = config.get("dygraph.train_data_dir", None)
    epochs = config.get("dygraph.epochs", None)
    feature_size = config.get('hyper_parameters.feature_size', None)
    print_interval = config.get("dygraph.print_interval", None)
    model_save_path = config.get("dygraph.model_save_path", "model_output")
    neg_num = config.get("hyper_parameters.neg_num")

    print("***********************************")
    logger.info(
        "use_gpu: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, train_data_dir, epochs, print_interval,
               model_save_path))
    print("***********************************")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    word2vec_model = create_model(config)

    model_init_path = config.get("dygraph.model_init_path", None)
    if model_init_path is not None:
        load_model(model_init_path, word2vec_model)

    # to do : add optimizer function
    optimizer = paddle.optimizer.Adam(parameters=word2vec_model.parameters())

    file_list = [
        os.path.join(train_data_dir, x) for x in os.listdir(train_data_dir)
    ]
    print("read data")
    dataset = Word2VecDataset(file_list, config)
    train_dataloader = create_data_loader(dataset, place=place, config=config)

    last_epoch_id = config.get("last_epoch", -1)

    for epoch_id in range(last_epoch_id + 1, epochs):
        # set train mode
        word2vec_model.train()
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

            input_word, true_word, neg_word = create_feeds(batch, neg_num)

            true_logits, neg_logits = word2vec_model.forward(
                [input_word, true_word, neg_word])
            loss = create_loss(true_logits, neg_logits, neg_num)

            loss.backward()
            optimizer.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size
            if batch_id % print_interval == 1:
                logger.info(
                    "epoch: {}, batch_id: {}, loss: {:.6f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(epoch_id, batch_id,
                           loss.numpy()[0], train_reader_cost / print_interval,
                           (train_reader_cost + train_run_cost
                            ) / print_interval, total_samples / print_interval,
                           total_samples / (train_reader_cost + train_run_cost
                                            )))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()

        logger.info(
            "epoch: {} done, loss: {:.6f}, : epoch time{:.2f} s".format(
                epoch_id, loss.numpy()[0], time.time() - epoch_begin))

        for prefix, layer in word2vec_model.named_sublayers():
            if prefix == 'embedding':
                save_model(
                    layer, optimizer, model_save_path, epoch_id, prefix='rec')


if __name__ == '__main__':
    args = parse_args()
    main(args)
