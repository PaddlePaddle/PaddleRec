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
import gate_dnn_net as net
import time
import logging

from utils import load_yaml, get_abs_model, save_model, load_model
from criteo_reader_dygraph import CriteoDataset
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


def create_feeds(batch, dense_feature_dim):
    sparse_tensor = []
    for b in batch[:-1]:
        sparse_tensor.append(
            paddle.to_tensor(b.numpy().astype('int64').reshape(-1, 1)))
    dense_tensor = paddle.to_tensor(batch[-1].numpy().astype('float32')
                                    .reshape(-1, dense_feature_dim))

    label = sparse_tensor[0]
    return label, sparse_tensor[1:], dense_tensor


def create_model(config):
    sparse_feature_number = config.get(
        "hyper_parameters.sparse_feature_number")
    sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
    fc_sizes = config.get("hyper_parameters.fc_sizes")
    sparse_fea_num = config.get('hyper_parameters.sparse_fea_num')
    dense_feature_dim = config.get('hyper_parameters.dense_input_dim')
    sparse_input_slot = config.get('hyper_parameters.sparse_inputs_slots')
    use_embedding_gate = config.get('hyper_parameters.use_embedding_gate')
    use_hidden_gate = config.get('hyper_parameters.use_hidden_gate')

    gate_dnn_model = net.GateDNNLayer(
        sparse_feature_number, sparse_feature_dim, dense_feature_dim,
        sparse_input_slot - 1, fc_sizes, use_embedding_gate, use_hidden_gate)

    return gate_dnn_model


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
    dense_input_dim = config.get('hyper_parameters.dense_input_dim', None)

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    print("***********************************")
    logger.info(
        "use_gpu: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, test_data_dir, start_epoch, end_epoch, print_interval,
               model_load_path))
    print("***********************************")

    gate_dnn_model = create_model(config)
    file_list = [
        os.path.join(test_data_dir, x) for x in os.listdir(test_data_dir)
    ]
    dataset = CriteoDataset(file_list)
    test_dataloader = create_data_loader(dataset, place=place, config=config)

    auc_metric = paddle.metric.Auc("ROC")
    epoch_begin = time.time()
    interval_begin = time.time()

    epoch_id = end_epoch - 1
    logger.info("load model epoch {}".format(epoch_id))
    model_path = os.path.join(model_load_path, str(epoch_id))
    load_model(model_path, gate_dnn_model)
    for batch_id, batch in enumerate(test_dataloader()):
        batch_size = len(batch[0])

        label, sparse_tensor, dense_tensor = create_feeds(batch,
                                                          dense_input_dim)

        raw_pred = gate_dnn_model(sparse_tensor, dense_tensor)
        predict_2d = paddle.concat(x=[1 - raw_pred, raw_pred], axis=1)
        auc_metric.update(preds=predict_2d.numpy(), labels=label.numpy())

        if batch_id % print_interval == 1:
            logger.info(
                "infer epoch: {}, batch_id: {}, auc: {:.6f}, speed: {:.2f} ins/s".
                format(epoch_id, batch_id,
                       auc_metric.accumulate(), print_interval * batch_size / (
                           time.time() - interval_begin)))
            interval_begin = time.time()

    logger.info(
        "infer epoch: {} done, auc: {:.6f}, : epoch time{:.2f} s".format(
            epoch_id, auc_metric.accumulate(), time.time() - epoch_begin))


if __name__ == '__main__':
    args = parse_args()
    main(args)
