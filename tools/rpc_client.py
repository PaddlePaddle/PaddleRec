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

import sys
import numpy as np
import argparse
import paddle
import os
import time
from paddle_serving_client import Client
from importlib import import_module
from paddle.io import DataLoader
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_config", type=str)
    parser.add_argument("--connect", type=str)
    parser.add_argument("--use_gpu", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--reader_file", type=str)
    parser.add_argument("--batchsize", type=int)
    args = parser.parse_args()
    args.use_gpu = (True if args.use_gpu.lower() == "true" else False)
    return args


def create_data_loader(args):
    data_dir = args.data_dir
    reader_path, reader_file = os.path.split(args.reader_file)
    reader_file, extension = os.path.splitext(reader_file)
    batchsize = args.batchsize
    place = args.place
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    sys.path.append(reader_path)
    reader_class = import_module(reader_file)
    dataset = reader_class.RecDataset(file_list, config=None)
    loader = DataLoader(
        dataset, batch_size=batchsize, places=place, drop_last=True)
    return loader


def run_client(args):
    client = Client()
    client.load_client_config(args.client_config)
    client.connect([args.connect])
    place = paddle.set_device('gpu' if args.use_gpu else 'cpu')
    args.place = place
    test_dataloader = create_data_loader(args)
    feed_names = client.feed_names_
    fetch_names = client.fetch_names_

    for batch_id, batch_data in enumerate(test_dataloader):
        batch_data = [tensor.numpy() for tensor in batch_data]
        feed_dict = dict(zip(feed_names, batch_data))
        fetch_map = client.predict(
            feed=feed_dict, fetch=fetch_names, batch=True)
        print(fetch_map)


if __name__ == '__main__':
    args = parse_args()
    run_client(args)
