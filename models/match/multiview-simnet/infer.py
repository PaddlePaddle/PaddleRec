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
from evaluate_reader_dygraph import BQDataset
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


def create_feeds(batch, query_len, pos_len):
    q_slots = [
        paddle.to_tensor(batch[0].numpy().astype('int64').reshape(-1,
                                                                  query_len))
    ]
    pt_slots = [
        paddle.to_tensor(batch[1].numpy().astype('int64').reshape(-1, pos_len))
    ]
    inputs = [q_slots, pt_slots]
    return inputs


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
    batch_size = config.get('dygraph.batch_size_infer', None)
    loader = DataLoader(dataset, batch_size=batch_size, places=place)
    return loader


def main(args):
    paddle.seed(12345)
    config = load_yaml(args.config_yaml)
    use_gpu = config.get("dygraph.use_gpu", False)
    test_data_dir = config.get("dygraph.test_data_dir", None)
    epochs = config.get("dygraph.epochs", None)
    print_interval = config.get("dygraph.print_interval", None)
    model_load_path = config.get("dygraph.infer_load_path",
                                 "increment_dygraph")
    start_epoch = config.get("dygraph.infer_start_epoch", -1)
    end_epoch = config.get("dygraph.infer_end_epoch", 1)
    batch_size = config.get('dygraph.batch_size_infer', None)
    margin = config.get('hyper_parameters.margin', 0.1)
    query_len = config.get('hyper_parameters.query_len', 79)
    pos_len = config.get('hyper_parameters.pos_len', 99)
    neg_len = config.get('hyper_parameters.neg_len', 90)

    print("***********************************")
    logger.info(
        "use_gpu: {}, test_data_dir: {}, epochs: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, test_data_dir, epochs, print_interval,
               model_load_path))
    print("***********************************")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    simnet_model = create_model(config)
    # to do init model
    file_list = [
        os.path.join(test_data_dir, x) for x in os.listdir(test_data_dir)
    ]
    print("read data")
    dataset = BQDataset(file_list)
    test_dataloader = create_data_loader(dataset, place=place, config=config)

    epoch_begin = time.time()
    interval_begin = time.time()

    for epoch_id in range(start_epoch + 1, end_epoch):

        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, simnet_model)

        for batch_id, batch in enumerate(test_dataloader()):

            inputs = create_feeds(batch, query_len, pos_len)

            cos_pos, cos_neg = simnet_model(inputs, True)

            if batch_id % print_interval == 0:
                logger.info(
                    "infer epoch: {}, batch_id: {}, query_pt_sim: {}, speed: {:.2f} ins/s".
                    format(epoch_id, batch_id,
                           cos_pos.numpy(), print_interval * batch_size / (
                               time.time() - interval_begin)))
                interval_begin = time.time()

        logger.info(
            "infer epoch: {} done, query_pt_sim: {}, : epoch time{:.2f} s".
            format(epoch_id, cos_pos.numpy(), time.time() - epoch_begin))


if __name__ == '__main__':
    args = parse_args()
    main(args)
