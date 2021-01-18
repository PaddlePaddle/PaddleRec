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
from synthetic_evaluate_reader_dygraph import SyntheticDataset
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
    return query, doc_pos


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
    test_data_dir = config.get("dygraph.test_data_dir", None)
    epochs = config.get("dygraph.epochs", None)
    print_interval = config.get("dygraph.print_interval", None)
    model_load_path = config.get("dygraph.infer_load_path",
                                 "increment_dygraph")
    trigram_d = config.get('hyper_parameters.trigram_d', None)
    start_epoch = config.get("dygraph.infer_start_epoch", -1)
    end_epoch = config.get("dygraph.infer_end_epoch", 1)

    print("***********************************")
    logger.info(
        "use_gpu: {}, test_data_dir: {}, epochs: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, test_data_dir, epochs, print_interval,
               model_load_path))
    print("***********************************")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    dssm_model = create_model(config)
    # to do init model
    file_list = [
        os.path.join(test_data_dir, x) for x in os.listdir(test_data_dir)
    ]
    print("read data")
    dataset = SyntheticDataset(file_list)
    test_dataloader = create_data_loader(dataset, place=place, config=config)

    epoch_begin = time.time()
    interval_begin = time.time()

    for epoch_id in range(start_epoch + 1, end_epoch):

        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, dssm_model)

        for batch_id, batch in enumerate(test_dataloader()):
            batch_size = len(batch[0])

            query, doc_pos = create_feeds(batch, trigram_d)

            inputs = [query, doc_pos]

            R_Q_D_p, hit_prob = dssm_model(inputs, True)

            if batch_id % print_interval == 0:
                logger.info(
                    "infer epoch: {}, batch_id: {}, query_doc_sim: {}, speed: {:.2f} ins/s".
                    format(epoch_id, batch_id,
                           R_Q_D_p.numpy(), print_interval * batch_size / (
                               time.time() - interval_begin)))
                interval_begin = time.time()

        logger.info(
            "infer epoch: {} done, query_doc_sim: {}, : epoch time{:.2f} s".
            format(epoch_id, R_Q_D_p.numpy(), time.time() - epoch_begin))


if __name__ == '__main__':
    args = parse_args()
    main(args)
