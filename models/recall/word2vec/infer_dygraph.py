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
import numpy as np

from utils_dygraph import load_yaml, get_abs_model, save_model, load_model
from word2vec_reader_dygraph import Word2VecInferDataset
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


def create_feeds(batch, vocab_size):
    all_label = paddle.to_tensor(np.arange(vocab_size).astype('int32'))
    inputs = [
        paddle.to_tensor(batch[i].numpy().astype('int32')) for i in range(4)
    ]
    inputs_word = batch[4].numpy()
    return inputs, all_label, inputs_word


def create_model(config):
    sparse_feature_number = config.get(
        "hyper_parameters.sparse_feature_number")
    sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")

    word2vec = net.Word2VecInferLayer(sparse_feature_number,
                                      sparse_feature_dim, "emb")

    return word2vec


def create_data_loader(dataset, place, config):
    batch_size = config.get('dygraph.batch_size', None)
    loader = DataLoader(
        dataset, batch_size=batch_size, places=place, drop_last=True)
    return loader


def main(args):
    paddle.seed(12345)
    config = load_yaml(args.config_yaml)
    use_gpu = config.get("dygraph.use_gpu", True)
    test_data_dir = config.get("dygraph.test_data_dir", None)
    feature_size = config.get('hyper_parameters.feature_size', None)
    print_interval = config.get("dygraph.print_interval", None)
    model_load_path = config.get("dygraph.infer_load_path", "model_output")
    start_epoch = config.get("dygraph.infer_start_epoch", -1)
    end_epoch = config.get("dygraph.infer_end_epoch", 10)
    vocab_size = config.get("hyper_parameters.sparse_feature_number", 10)

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    print("***********************************")
    logger.info(
        "use_gpu: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, test_data_dir, start_epoch, end_epoch, print_interval,
               model_load_path))
    print("***********************************")

    word2vec_model = create_model(config)
    file_list = [
        os.path.join(test_data_dir, x) for x in os.listdir(test_data_dir)
    ]
    print("read data")
    dataset = Word2VecInferDataset(file_list, config)
    test_dataloader = create_data_loader(dataset, place=place, config=config)

    auc_metric = paddle.metric.Auc("ROC")
    epoch_begin = time.time()
    interval_begin = time.time()

    for epoch_id in range(start_epoch + 1, end_epoch):

        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, word2vec_model)
        accum_num_sum = 0
        accum_num = 0
        for batch_id, batch in enumerate(test_dataloader()):
            batch_size = len(batch[0])

            inputs, all_label, inputs_word = create_feeds(batch, vocab_size)
            label = inputs[3].numpy()
            val, pred_idx = word2vec_model(inputs[0], inputs[1], inputs[2],
                                           inputs[3], all_label)
            pre = pred_idx.numpy()

            for ii in range(len(label)):
                top4 = pre[ii][0]
                accum_num_sum += 1
                for idx in top4:
                    if int(idx) in inputs_word[ii]:
                        continue
                    print(int(idx), int(label[ii][0]))
                    if int(idx) == int(label[ii][0]):
                        accum_num += 1
                    break

            if batch_id % print_interval == 1:
                logger.info(
                    "infer epoch: {}, batch_id: {}, acc: {:.6f}, speed: {:.2f} ins/s".
                    format(epoch_id, batch_id, accum_num * 1.0 / accum_num_sum,
                           print_interval * batch_size / (time.time() -
                                                          interval_begin)))
                interval_begin = time.time()

        logger.info(
            "infer epoch: {} done, auc: {:.6f}, : epoch time{:.2f} s".format(
                epoch_id, auc_metric.accumulate(), time.time() - epoch_begin))


if __name__ == '__main__':
    args = parse_args()
    main(args)
