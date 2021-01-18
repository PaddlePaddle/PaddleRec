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
import recall_net as net
import time
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


def create_loss(predict, label_input):
    cost = F.square_error_cost(
        predict, paddle.cast(
            x=label_input, dtype='float32'))
    avg_cost = paddle.mean(cost)

    return avg_cost


def create_model(config):
    sparse_feature_number = config.get(
        "hyper_parameters.sparse_feature_number")
    sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
    fc_sizes = config.get("hyper_parameters.fc_sizes")

    Recall = net.DNNLayer(sparse_feature_number, sparse_feature_dim, fc_sizes)

    return Recall


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
    batch_size = config.get("dygraph.batch_size", 128)

    print("***********************************")
    logger.info(
        "use_gpu: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, train_data_dir, epochs, print_interval,
               model_save_path))
    print("***********************************")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    recall_model = create_model(config)
    model_init_path = config.get("dygraph.model_init_path", None)
    if model_init_path is not None:
        load_model(model_init_path, recall_model)

    # to do : add optimizer function
    optimizer = paddle.optimizer.Adam(parameters=recall_model.parameters())
    filelist = os.listdir(train_data_dir)
    filelist.sort()
    file_list = [os.path.join(train_data_dir, x) for x in filelist]
    print("read data")
    dataset = MovieDataset(file_list)
    train_dataloader = create_data_loader(dataset, place=place, config=config)

    last_epoch_id = config.get("last_epoch", -1)

    for epoch_id in range(last_epoch_id + 1, epochs):
        # set train mode
        recall_model.train()
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
            # batch_size = len(batch[0])
            batch_size = config.get("dygraph.batch_size", 128)

            user_sparse_inputs, mov_sparse_inputs, label_input = create_feeds(
                batch)

            predict = recall_model(batch_size, user_sparse_inputs,
                                   mov_sparse_inputs, label_input)

            loss = create_loss(predict, label_input)

            loss.backward()
            optimizer.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size

            if batch_id % print_interval == 0:
                logger.info(
                    "epoch: {}, batch_id: {}, LOSS: {}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
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

        logger.info("epoch: {} done, epoch time: {:.2f} s".format(
            epoch_id, time.time() - epoch_begin))

        save_model(
            recall_model, optimizer, model_save_path, epoch_id, prefix='rec')


if __name__ == '__main__':
    args = parse_args()
    main(args)
