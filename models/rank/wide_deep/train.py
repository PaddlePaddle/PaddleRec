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
import paddle.nn.functional as F
import os
import paddle.nn as nn
import wide_deep_net as net
import time
import logging

from utils import load_yaml, get_abs_model, save_model, load_model
from reader_dygraph import WideDeepDataset
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


def create_feeds(batch, wide_input_dim, deep_input_dim):
    sparse_tensor = []
    label = paddle.to_tensor(batch[0].numpy().astype('int64').reshape(-1, 1))
    wide_tensor = paddle.to_tensor(batch[1].numpy().astype('float32').reshape(
        -1, wide_input_dim))
    deep_tensor = paddle.to_tensor(batch[2].numpy().astype('float32').reshape(
        -1, deep_input_dim))
    return label, wide_tensor, deep_tensor


def create_loss(prediction, label):
    pred = F.sigmoid(prediction)
    cost = paddle.nn.functional.log_loss(
        input=pred, label=paddle.cast(
            label, dtype="float32"))
    avg_cost = paddle.mean(x=cost)
    return avg_cost


def create_model(config):
    wide_input_dim = config.get('hyper_parameters.wide_input_dim')
    deep_input_dim = config.get('hyper_parameters.deep_input_dim')
    hidden1_units = config.get("hyper_parameters.hidden1_units")
    hidden2_units = config.get("hyper_parameters.hidden2_units")
    hidden3_units = config.get("hyper_parameters.hidden3_units")

    layer_sizes = [hidden1_units, hidden2_units, hidden3_units]
    wide_deep_model = net.WideDeepLayer(wide_input_dim, deep_input_dim,
                                        layer_sizes)

    return wide_deep_model


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
    print_interval = config.get("dygraph.print_interval", None)
    model_save_path = config.get("dygraph.model_save_path", "model_output")
    wide_input_dim = config.get('hyper_parameters.wide_input_dim', None)
    deep_input_dim = config.get('hyper_parameters.deep_input_dim', None)

    print("***********************************")
    logger.info(
        "use_gpu: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, train_data_dir, epochs, print_interval,
               model_save_path))
    print("***********************************")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    wide_deep_model = create_model(config)
    model_init_path = config.get("dygraph.model_init_path", None)
    if model_init_path is not None:
        load_model(model_init_path, wide_deep_model)

    # to do : add optimizer function
    optimizer = paddle.optimizer.Adam(parameters=wide_deep_model.parameters())

    file_list = [
        os.path.join(train_data_dir, x) for x in os.listdir(train_data_dir)
    ]
    print("read data")
    dataset = WideDeepDataset(file_list)
    train_dataloader = create_data_loader(dataset, place=place, config=config)

    last_epoch_id = config.get("last_epoch", -1)

    for epoch_id in range(last_epoch_id + 1, epochs):
        # set train mode
        wide_deep_model.train()
        auc_metric = paddle.metric.Auc("ROC")
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

            label, wide_tensor, deep_tensor = create_feeds(
                batch, wide_input_dim, deep_input_dim)

            prediction = wide_deep_model.forward(wide_tensor, deep_tensor)
            loss = create_loss(prediction, label)

            loss.backward()
            optimizer.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size
            pred = paddle.nn.functional.sigmoid(
                paddle.clip(
                    prediction, min=-15.0, max=15.0),
                name="prediction")
            label_int = paddle.cast(label, 'int64')

            # for acc
            correct = acc_metric.compute(pred, label_int)
            acc_metric.update(correct)
            # for auc
            predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
            auc_metric.update(
                preds=predict_2d.numpy(), labels=label_int.numpy())

            if batch_id % print_interval == 1:
                logger.info(
                    "epoch: {}, batch_id: {}, auc: {:.6f}, acc: {:.5f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(epoch_id, batch_id,
                           auc_metric.accumulate(),
                           acc_metric.accumulate(), train_reader_cost /
                           print_interval, (train_reader_cost + train_run_cost
                                            ) / print_interval, total_samples /
                           print_interval, total_samples / (train_reader_cost +
                                                            train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()

        logger.info(
            "epoch: {} done, auc: {:.6f}, acc: {:.6f}, : epoch time{:.2f} s".
            format(epoch_id,
                   auc_metric.accumulate(),
                   acc_metric.accumulate(), time.time() - epoch_begin))

        save_model(
            wide_deep_model,
            optimizer,
            model_save_path,
            epoch_id,
            prefix='rec')


if __name__ == '__main__':
    args = parse_args()
    main(args)
