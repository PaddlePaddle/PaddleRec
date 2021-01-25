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

from utils import load_yaml, get_abs_model, save_model, load_model
from NAMLDataReader import RecDataset
from paddle.io import DistributedBatchSampler, DataLoader
import argparse
import net
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
    # for b in batch[:-1]:
    #     sparse_tensor.append(
    #         paddle.to_tensor(b.numpy().astype('int64').reshape(-1, 1)))
    # dense_tensor = paddle.to_tensor(batch[-1].numpy().astype('float32')
    #                                 .reshape(-1, dense_feature_dim))

    label = batch[0]
    return label, batch[1:], None


def create_loss(raw_predict_2d, label):
    cost = paddle.nn.functional.cross_entropy(
        input=raw_predict_2d, label=label)
    avg_cost = paddle.mean(x=cost)

    return avg_cost


def create_model(config):

    return net.NAMLLayer(config)


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
    dense_input_dim = config.get('hyper_parameters.dense_input_dim', None)

    print("***********************************")
    logger.info(
        "use_gpu: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, train_data_dir, epochs, print_interval,
               model_save_path))
    print("***********************************")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    dnn_model = create_model(config)
    model_init_path = config.get("dygraph.model_init_path", None)
    if model_init_path is not None:
        load_model(model_init_path, dnn_model)

    optimizer = paddle.optimizer.Adam(parameters=dnn_model.parameters())

    file_list = [
        os.path.join(train_data_dir, x) for x in os.listdir(train_data_dir)
    ]

    dataset = RecDataset(file_list,config)
    train_dataloader = create_data_loader(dataset, place=place, config=config)

    last_epoch_id = -1

    for epoch_id in range(last_epoch_id + 1, epochs):
        dnn_model.train()
        auc_metric = paddle.metric.Auc("ROC")
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

            label, sparse_tensor, dense_tensor = create_feeds(batch,
                                                              dense_input_dim)

            labels = paddle.cast(label, "float32")
            #print(labels)
            #sampe_size = config.get("hyper_parameters.neg_condidate_sample_size") + 1
            #label_mat = paddle.reshape(label,[-1,1,sampe_size])
            cate_sample, cate_visit, sub_cate_sample, sub_cate_visit, title_sample, title_visit, content_sample, content_visit = sparse_tensor[:]
            # cate_sample = paddle.reshape(cate_sample,[-1,sampe_size,1])
            # cate_sample = paddle.matmul(label_mat, cate_sample)
            # cate_sample = paddle.reshape(cate_sample,[-1])
            # sub_cate_sample = paddle.reshape(sub_cate_sample,[-1,sampe_size,1])
            # [b,1,1]
            # sub_cate_sample = paddle.matmul(label_mat, sub_cate_sample)
            # sub_cate_sample = paddle.reshape(sub_cate_sample, [-1])
            raw_pred = dnn_model(cate_sample, cate_visit, sub_cate_sample, sub_cate_visit, title_sample, title_visit, content_sample, content_visit)
            loss = paddle.nn.functional.cross_entropy(input = raw_pred, label = labels, soft_label=True)
            loss = paddle.mean(loss)
            loss.backward()
            optimizer.step()
            train_run_cost += time.time() - train_start
            total_samples += batch_size
        #     predict_2d = paddle.concat(x=[1 - raw_pred, raw_pred], axis=1)
#            auc_metric.update(preds=raw_pred.numpy(), labels=label.numpy())

            if batch_id % print_interval == 0:
                print(raw_pred)
                print(label)
                # logger.info(
                #     "epoch: {}, batch_id: {}, auc: {:.6f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} images/sec".
                #     format(epoch_id, batch_id,
                #            auc_metric.accumulate(), train_reader_cost /
                #            print_interval, (train_reader_cost + train_run_cost
                #                             ) / print_interval, total_samples /
                #            print_interval, total_samples / (train_reader_cost +
                #                                             train_run_cost)))
                # train_reader_cost = 0.0
                # train_run_cost = 0.0
                # total_samples = 0
                print(loss)
            reader_start = time.time()

        # logger.info("epoch: {} done, auc: {:.6f}, : epoch time{:.2f} s".format(
        #     epoch_id, auc_metric.accumulate(), time.time() - epoch_begin))
        #
        # save_model(
        #     dnn_model, optimizer, model_save_path, epoch_id, prefix='rec')


if __name__ == '__main__':
    from paddle.nn import Conv1D
    import numpy as np

    x = np.array([[[4, 8, 1, 9],
                   [7, 2, 0, 9],
                   [6, 9, 2, 6]]]).astype(np.float32)
    w = np.array(
        [[[9, 3, 4],
          [0, 0, 7],
          [2, 5, 6]],
         [[0, 3, 4],
          [2, 9, 7],
          [5, 6, 8]]]).astype(np.float32)
    x_t = paddle.to_tensor(x)
    conv = Conv1D(3, 2, 3, padding = "same")
    conv.weight.set_value(w)
    y_t = conv(x_t)
    print(y_t)
    # [[[133. 238.]
    #   [160. 211.]]]
    args = parse_args()
    main(args)
