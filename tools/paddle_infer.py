# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import time
import logging
import sys
from importlib import import_module
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from utils.utils_single import load_yaml, load_dy_model_class, get_abs_model
from utils.save_load import save_model, load_model
from paddle.io import DistributedBatchSampler, DataLoader
import argparse
from paddle.inference import Config
from paddle.inference import create_predictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--use_gpu", type=bool)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--reader_file", type=str)
    parser.add_argument("--batchsize", type=int)
    args = parser.parse_args()
    return args


def init_predictor(args):
    if args.model_dir:
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    if args.use_gpu:
        config.enable_use_gpu(1000, 0)
    else:
        config.disable_gpu()
    predictor = create_predictor(config)
    return predictor


def create_data_loader(args):
    data_dir = args.data_dir
    reader_file = args.reader_file.split(".")[0]
    batchsize = args.batchsize
    place = args.place
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    sys.path.append(os.path.abspath("."))
    reader_class = import_module(reader_file)
    dataset = reader_class.RecDataset(file_list, config=None)
    loader = DataLoader(
        dataset, batch_size=batchsize, places=place, drop_last=True)
    return loader


def main(args):
    predictor = init_predictor(args)
    place = paddle.set_device('gpu' if args.use_gpu else 'cpu')
    args.place = place
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    test_dataloader = create_data_loader(args)
    for batch_id, batch_data in enumerate(test_dataloader):
        name_data_pair = dict(zip(input_names, batch_data))
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(name_data_pair[name].numpy())
        predictor.run()
        results = []
        for name in output_names:
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()[0]
            results.append(output_data)
        print(results)


if __name__ == '__main__':
    args = parse_args()
    main(args)
