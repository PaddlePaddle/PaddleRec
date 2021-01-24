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
import time
import logging
import sys
import importlib

import net
import numpy as np
__dir__ = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from utils.utils_single import load_yaml, load_dy_model_class, get_abs_model, create_data_loader
from utils.save_load import save_model, load_model
from paddle.io import DistributedBatchSampler, DataLoader
import argparse

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--config_yaml", type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def create_feeds(batch_data, vocab_size):
    all_label = paddle.to_tensor(np.arange(vocab_size).astype('int32'))
    inputs = [
        paddle.to_tensor(batch_data[i].numpy().astype('int32'))
        for i in range(4)
    ]
    inputs_word = batch_data[4].numpy()
    return inputs, all_label, inputs_word


def create_model(config):
    sparse_feature_number = config.get(
        "hyper_parameters.sparse_feature_number")
    sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")

    word2vec = net.Word2VecInferLayer(sparse_feature_number,
                                      sparse_feature_dim, "emb")

    return word2vec


def main(args):
    paddle.seed(12345)
    # load config
    config = load_yaml(args.config_yaml)
    dy_model_class = load_dy_model_class(args.abs_dir)
    config["config_abs_dir"] = args.abs_dir
    # tools.vars
    use_gpu = config.get("runner.use_gpu", True)
    test_data_dir = config.get("runner.test_data_dir", None)
    print_interval = config.get("runner.print_interval", None)
    model_load_path = config.get("runner.infer_load_path", "model_output")
    start_epoch = config.get("runner.infer_start_epoch", 0)
    end_epoch = config.get("runner.infer_end_epoch", 10)
    vocab_size = config.get("hyper_parameters.sparse_feature_number", 10)

    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, test_data_dir, start_epoch, end_epoch, print_interval,
               model_load_path))
    logger.info("**************common.configs**********")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    #dy_model = dy_model_class.create_model(config)
    dy_model = create_model(config)

    # to do : add optimizer function
    #optimizer = dy_model_class.create_optimizer(dy_model, config)

    logger.info("read data")
    test_dataloader = create_data_loader(
        config=config, place=place, mode="test")

    epoch_begin = time.time()
    interval_begin = time.time()

    metric_list, metric_list_name = dy_model_class.create_metrics()

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, dy_model)
        dy_model.eval()
        accum_num_sum = 0
        accum_num = 0
        for batch_id, batch in enumerate(test_dataloader()):
            batch_size = len(batch[0])

            inputs, all_label, inputs_word = create_feeds(batch, vocab_size)
            label = inputs[3].numpy()
            val, pred_idx = dy_model.forward(inputs[0], inputs[1], inputs[2],
                                             all_label)
            pre = pred_idx.numpy()

            for ii in range(len(label)):
                top4 = pre[ii][0]
                accum_num_sum += 1
                for idx in top4:
                    if int(idx) in inputs_word[ii]:
                        continue
                    if int(idx) == int(label[ii][0]):
                        accum_num += 1
                    break

            if batch_id % print_interval == 0:
                logger.info(
                    "infer epoch: {}, batch_id: {}, acc: {:.6f}, speed: {:.2f} ins/s".
                    format(epoch_id, batch_id, accum_num * 1.0 / accum_num_sum,
                           print_interval * batch_size / (time.time() -
                                                          interval_begin)))
                interval_begin = time.time()

        logger.info("infer epoch: {} done, acc: {:.6f}, : epoch time{:.2f} s".
                    format(epoch_id, accum_num * 1.0 / accum_num_sum,
                           time.time() - epoch_begin))

        epoch_begin = time.time()


if __name__ == '__main__':
    args = parse_args()
    main(args)
