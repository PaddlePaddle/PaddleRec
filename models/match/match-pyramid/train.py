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
import pyramid_net as net
import time
import logging
import paddle.nn.functional as F
from utils import load_yaml, get_abs_model, save_model, load_model
from reader_dygraph import LetorDataset
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


def create_feeds(batch, sentence_left_size, sentence_right_size):
    sentence_left = paddle.to_tensor(batch[0].numpy().astype('int64').reshape(
        -1, sentence_left_size))
    sentence_right = paddle.to_tensor(batch[1].numpy().astype('int64').reshape(
        -1, sentence_right_size))

    return [sentence_left, sentence_right]


def create_loss(prediction):
    pos = paddle.slice(prediction, axes=[0, 1], starts=[0, 0], ends=[64, 1])
    neg = paddle.slice(prediction, axes=[0, 1], starts=[64, 0], ends=[128, 1])
    loss_part1 = paddle.subtract(
        paddle.full(
            shape=[64, 1], fill_value=1.0, dtype='float32'), pos)
    loss_part2 = paddle.add(loss_part1, neg)
    loss_part3 = paddle.maximum(
        paddle.full(
            shape=[64, 1], fill_value=0.0, dtype='float32'),
        loss_part2)

    avg_cost = paddle.mean(loss_part3)
    return avg_cost


def create_model(config):
    emb_path = config.get("hyper_parameters.emb_path")
    vocab_size = config.get("hyper_parameters.vocab_size")
    emb_size = config.get("hyper_parameters.emb_size")
    kernel_num = config.get("hyper_parameters.kernel_num")
    conv_filter = config.get("hyper_parameters.conv_filter")
    conv_act = config.get("hyper_parameters.conv_act")
    hidden_size = config.get("hyper_parameters.hidden_size")
    out_size = config.get("hyper_parameters.out_size")
    pool_size = config.get("hyper_parameters.pool_size")
    pool_stride = config.get("hyper_parameters.pool_stride")
    pool_padding = config.get("hyper_parameters.pool_padding")
    pool_type = config.get("hyper_parameters.pool_type")
    hidden_act = config.get("hyper_parameters.hidden_act")

    pyramid_model = net.MatchPyramidLayer(
        emb_path, vocab_size, emb_size, kernel_num, conv_filter, conv_act,
        hidden_size, out_size, pool_size, pool_stride, pool_padding, pool_type,
        hidden_act)

    return pyramid_model


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
    sentence_left_size = config.get("hyper_parameters.sentence_left_size")
    sentence_right_size = config.get("hyper_parameters.sentence_right_size")

    print("***********************************")
    logger.info(
        "use_gpu: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, train_data_dir, epochs, print_interval,
               model_save_path))
    print("***********************************")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    pyramid_model = create_model(config)
    model_init_path = config.get("dygraph.model_init_path", None)
    if model_init_path is not None:
        load_model(model_init_path, pyramid_model)

    # to do : add optimizer function
    optimizer = paddle.optimizer.Adam(parameters=pyramid_model.parameters())
    filelist = os.listdir(train_data_dir)
    filelist.sort()
    file_list = [os.path.join(train_data_dir, x) for x in filelist]
    print("read data")
    dataset = LetorDataset(file_list)
    train_dataloader = create_data_loader(dataset, place=place, config=config)

    last_epoch_id = config.get("last_epoch", -1)

    for epoch_id in range(last_epoch_id + 1, epochs):
        # set train mode
        pyramid_model.train()
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

            inputs = create_feeds(batch, sentence_left_size,
                                  sentence_right_size)

            prediction = pyramid_model(inputs)

            loss = create_loss(prediction)

            loss.backward()
            optimizer.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size

            if batch_id % print_interval == 0:
                logger.info(
                    "epoch: {}, batch_id: {}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(epoch_id, batch_id, train_reader_cost /
                           print_interval, (train_reader_cost + train_run_cost
                                            ) / print_interval, total_samples /
                           print_interval, total_samples / (train_reader_cost +
                                                            train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()

        logger.info("epoch: {} done, epoch time: {:.2f} s".format(
            epoch_id, time.time() - epoch_begin))

        save_model(
            pyramid_model, optimizer, model_save_path, epoch_id, prefix='rec')


if __name__ == '__main__':
    args = parse_args()
    main(args)
