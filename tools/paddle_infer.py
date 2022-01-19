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
import re
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
    parser.add_argument("--use_gpu", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--reader_file", type=str)
    parser.add_argument("--batchsize", type=int)
    parser.add_argument("--model_name", type=str, default="rec_model")
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--enable_mkldnn", type=str, default="False")
    parser.add_argument("--enable_tensorRT", type=str, default="False")
    parser.add_argument("--benchmark", type=str, default="True")
    parser.add_argument("--save_log_path", type=str, default="./output")
    parser.add_argument("--precision", type=str)
    args = parser.parse_args()
    args.use_gpu = (True if args.use_gpu.lower() == "true" else False)
    args.enable_mkldnn = (True
                          if args.enable_mkldnn.lower() == "true" else False)
    args.enable_tensorRT = (True if args.enable_tensorRT.lower() == "true" else
                            False)
    args.benchmark = (True if args.benchmark.lower() == "true" else False)
    return args


def init_predictor(args):
    if args.model_dir:
        has_model = 0
        pdmodel_name = 0
        pdiparams_name = 0
        for file_name in os.listdir(args.model_dir):
            if re.search("__model__", file_name):
                has_model = 1
            if file_name.endswith(".pdmodel"):
                pdmodel_name = os.path.join(args.model_dir, file_name)
            if file_name.endswith(".pdiparams"):
                pdiparams_name = os.path.join(args.model_dir, file_name)
        if has_model == 1:
            config = Config(args.model_dir)
        elif pdmodel_name and pdiparams_name:
            config = Config(pdmodel_name, pdiparams_name)
        else:
            raise ValueError(
                "config setting error, please check your model path")
    else:
        config = Config(args.model_file, args.params_file)

    if args.use_gpu:
        config.enable_use_gpu(1000, 0)
        if args.enable_tensorRT:
            config.enable_tensorrt_engine(
                max_batch_size=args.batchsize,
                min_subgraph_size=9,
                precision_mode=paddle.inference.PrecisionType.Float32)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            config.enable_mkldnn()
    predictor = create_predictor(config)
    return predictor, config


def create_data_loader(args):
    data_dir = args.data_dir
    reader_path, reader_file = os.path.split(args.reader_file)
    reader_file, extension = os.path.splitext(reader_file)
    batchsize = args.batchsize
    place = args.place
    file_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    sys.path.append(reader_path)
    #sys.path.append(os.path.abspath("."))
    reader_class = import_module(reader_file)
    config = {"runner.inference": True}
    dataset = reader_class.RecDataset(file_list, config=config)
    loader = DataLoader(
        dataset, batch_size=batchsize, places=place, drop_last=True)
    return loader


def main(args):
    predictor, pred_config = init_predictor(args)
    place = paddle.set_device('gpu' if args.use_gpu else 'cpu')
    args.place = place
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    test_dataloader = create_data_loader(args)

    if args.benchmark:
        import auto_log
        pid = os.getpid()
        autolog = auto_log.AutoLogger(
            model_name=args.model_name,
            model_precision=args.precision,
            batch_size=args.batchsize,
            data_shape="dynamic",
            save_path=args.save_log_path,
            inference_config=pred_config,
            pids=pid,
            process_name=None,
            gpu_ids=0,
            time_keys=[
                'preprocess_time', 'inference_time', 'postprocess_time'
            ])

    for batch_id, batch_data in enumerate(test_dataloader):
        name_data_pair = dict(zip(input_names, batch_data))
        if args.benchmark:
            autolog.times.start()
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(name_data_pair[name].numpy())
        if args.benchmark:
            autolog.times.stamp()
        predictor.run()
        for name in output_names:
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
        results = []
        results_type = []
        if args.benchmark:
            autolog.times.stamp()
        for name in output_names:
            results_type.append(output_tensor.type())
            results.append(output_data[0])
        if args.benchmark:
            autolog.times.end(stamp=True)
        print(results)

    if args.benchmark:
        autolog.report()


if __name__ == '__main__':
    args = parse_args()
    main(args)
