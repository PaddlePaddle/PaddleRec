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
import esmm_net as net
import time
import logging

from utils import load_yaml, get_abs_model, save_model, load_model
from esmm_reader_dygraph import ESMMDataset
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


def create_feeds(batch, max_len):
    sparse_tensor = []
    for b in batch[:-2]:
        sparse_tensor.append(
            paddle.to_tensor(b.numpy().astype('int64').reshape(-1, max_len)))
    ctr_label = paddle.to_tensor(batch[-2].numpy().astype('int64').reshape(-1,
                                                                           1))
    ctcvr_label = paddle.to_tensor(batch[-1].numpy().astype('int64').reshape(
        -1, 1))
    return sparse_tensor, ctr_label, ctcvr_label


def create_loss(ctr_out_one, ctr_clk, ctcvr_prop_one, ctcvr_buy):
    loss_ctr = paddle.nn.functional.log_loss(
        input=ctr_out_one, label=paddle.cast(
            ctr_clk, dtype="float32"))
    loss_ctcvr = paddle.nn.functional.log_loss(
        input=ctcvr_prop_one, label=paddle.cast(
            ctcvr_buy, dtype="float32"))
    cost = loss_ctr + loss_ctcvr
    avg_cost = paddle.mean(x=cost)
    return avg_cost


def create_model(config):
    max_len = config.get("hyper_parameters.max_len", 3)
    sparse_feature_number = config.get(
        "hyper_parameters.sparse_feature_number")
    sparse_feature_dim = config.get("hyper_parameters.sparse_feature_dim")
    num_field = config.get("hyper_parameters.num_field")
    learning_rate = config.get("hyper_parameters.optimizer.learning_rate")
    ctr_fc_sizes = config.get("hyper_parameters.ctr_fc_sizes")
    cvr_fc_sizes = config.get("hyper_parameters.cvr_fc_sizes")
    sparse_feature_number = config.get(
        "hyper_parameters.sparse_feature_number")

    esmm_model = net.ESMMLayer(sparse_feature_number, sparse_feature_dim,
                               num_field, ctr_fc_sizes, cvr_fc_sizes)

    return esmm_model


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
    max_len = config.get("hyper_parameters.max_len", 3)

    print("***********************************")
    logger.info(
        "use_gpu: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, train_data_dir, epochs, print_interval,
               model_save_path))
    print("***********************************")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    esmm_model = create_model(config)
    model_init_path = config.get("dygraph.model_init_path", None)
    if model_init_path is not None:
        load_model(model_init_path, esmm_model)

    # to do : add optimizer function
    optimizer = paddle.optimizer.Adam(parameters=esmm_model.parameters())

    file_list = [
        os.path.join(train_data_dir, x) for x in os.listdir(train_data_dir)
    ]
    print("read data")
    dataset = ESMMDataset(file_list)
    train_dataloader = create_data_loader(dataset, place=place, config=config)

    last_epoch_id = config.get("last_epoch", -1)

    for epoch_id in range(last_epoch_id + 1, epochs):
        # set train mode
        esmm_model.train()
        ctr_auc_metric = paddle.metric.Auc("ROC")
        ctcvr_auc_metric = paddle.metric.Auc("ROC")
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

            sparse_tensor, label_ctr, label_ctcvr = create_feeds(batch,
                                                                 max_len)

            ctr_out, ctr_out_one, cvr_out, cvr_out_one, ctcvr_prop, ctcvr_prop_one = esmm_model.forward(
                sparse_tensor)
            loss = create_loss(ctr_out_one, label_ctr, ctcvr_prop_one,
                               label_ctcvr)

            loss.backward()
            optimizer.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size

            # for auc
            ctr_auc_metric.update(
                preds=ctr_out.numpy(), labels=label_ctr.numpy())
            ctcvr_auc_metric.update(
                preds=ctcvr_prop.numpy(), labels=label_ctcvr.numpy())

            if batch_id % print_interval == 0:
                logger.info(
                    "epoch: {}, batch_id: {}, ctr_auc: {:.6f}, ctcvr_auc: {:.6f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                    format(epoch_id, batch_id,
                           ctr_auc_metric.accumulate(),
                           ctcvr_auc_metric.accumulate(), train_reader_cost /
                           print_interval, (train_reader_cost + train_run_cost
                                            ) / print_interval, total_samples /
                           print_interval, total_samples / (train_reader_cost +
                                                            train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            reader_start = time.time()

        logger.info(
            "epoch: {} done, ctr_auc: {:.6f}, ctcvr_auc: {:.6f}, epoch time:{:.2f} s".
            format(epoch_id,
                   ctr_auc_metric.accumulate(),
                   ctcvr_auc_metric.accumulate(), time.time() - epoch_begin))

        save_model(
            esmm_model, optimizer, model_save_path, epoch_id, prefix='rec')


if __name__ == '__main__':
    args = parse_args()
    main(args)
