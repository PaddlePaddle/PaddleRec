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
import textcnn_net as net
import time
import logging

from utils import load_yaml, get_abs_model, save_model, load_model
from reader_dygraph import TextCNNDataset
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
    input_data = paddle.to_tensor(batch[0].numpy().astype('int64').reshape(
        -1, 100))
    label = paddle.to_tensor(batch[1].numpy().astype('int64').reshape(-1, 1))

    return input_data, label


def create_loss(pred, label):
    cost = paddle.nn.functional.cross_entropy(input=pred, label=label)
    avg_cost = paddle.mean(x=cost)

    return avg_cost


def create_model(config):
    dict_dim = config.get("hyper_parameters.dict_dim")
    max_len = config.get("hyper_parameters.max_len")
    cnn_dim = config.get("hyper_parameters.cnn_dim")
    cnn_filter_size1 = config.get("hyper_parameters.cnn_filter_size1")
    cnn_filter_size2 = config.get("hyper_parameters.cnn_filter_size2")
    cnn_filter_size3 = config.get("hyper_parameters.cnn_filter_size3")
    filter_sizes = [cnn_filter_size1, cnn_filter_size2, cnn_filter_size3]
    emb_dim = config.get("hyper_parameters.emb_dim")
    hid_dim = config.get("hyper_parameters.hid_dim")
    class_dim = config.get("hyper_parameters.class_dim")
    is_sparse = config.get("hyper_parameters.is_sparse")

    textcnn_model = net.TextCNNLayer(
        dict_dim,
        emb_dim,
        class_dim,
        cnn_dim=cnn_dim,
        filter_sizes=filter_sizes,
        hidden_size=hid_dim)

    return textcnn_model


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
    num_field = config.get('hyper_parameters.num_field', None)

    print("***********************************")
    logger.info(
        "use_gpu: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, train_data_dir, epochs, print_interval,
               model_save_path))
    print("***********************************")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    textcnn_model = create_model(config)
    model_init_path = config.get("dygraph.model_init_path", None)
    if model_init_path is not None:
        load_model(model_init_path, textcnn_model)

    # to do : add optimizer function
    optimizer = paddle.optimizer.Adam(parameters=textcnn_model.parameters())

    file_list = [
        os.path.join(train_data_dir, x) for x in os.listdir(train_data_dir)
    ]
    print("read data")
    dataset = TextCNNDataset(file_list)
    train_dataloader = create_data_loader(dataset, place=place, config=config)

    last_epoch_id = config.get("last_epoch", -1)

    for epoch_id in range(last_epoch_id + 1, epochs):
        # set train mode
        textcnn_model.train()
        acc_metric = paddle.metric.Accuracy()
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

            input_data, label = create_feeds(batch)

            pred = textcnn_model.forward(input_data)
            loss = create_loss(pred, label)

            loss.backward()
            optimizer.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size
            # for acc
            prediction = paddle.nn.functional.softmax(pred)
            correct = acc_metric.compute(prediction, label)
            acc_metric.update(correct)

            if batch_id % print_interval == 1:
                logger.info(
                    "epoch: {}, batch_id: {}, acc: {:.6f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(epoch_id, batch_id,
                           acc_metric.accumulate(), train_reader_cost /
                           print_interval, (train_reader_cost + train_run_cost
                                            ) / print_interval, total_samples /
                           print_interval, total_samples / (train_reader_cost +
                                                            train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()

        logger.info("epoch: {} done, acc: {:.6f}, : epoch time{:.2f} s".format(
            epoch_id, acc_metric.accumulate(), time.time() - epoch_begin))

        save_model(
            textcnn_model, optimizer, model_save_path, epoch_id, prefix='rec')


if __name__ == '__main__':
    args = parse_args()
    main(args)
