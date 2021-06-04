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
from utils.benchmark_utils import PaddleInferBenchmark
from paddle.io import DistributedBatchSampler, DataLoader
import argparse
from paddle.inference import Config
from paddle.inference import create_predictor
import pynvml
import psutil
import GPUtil


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
    args = parser.parse_args()
    args.use_gpu = (True if args.use_gpu.lower() == "true" else False)
    args.enable_mkldnn = (True
                          if args.enable_mkldnn.lower() == "true" else False)
    args.enable_tensorRT = (True if args.enable_tensorRT.lower() == "true" else
                            False)
    return args


def init_predictor(args):
    if args.model_dir:
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    if args.use_gpu:
        config.enable_use_gpu(1000, 0)
        if args.enable_tensorRT:
            config.enable_tensorrt_engine(
                max_batch_size=args.batchsize,
                min_subgraph_size=1,
                precision_mode=paddle.inference.PrecisionType.Float32)
    else:
        config.disable_gpu()
        config.delete_pass("repeated_fc_relu_fuse_pass")
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
    dataset = reader_class.RecDataset(file_list, config=None)
    loader = DataLoader(
        dataset, batch_size=batchsize, places=place, drop_last=True)
    return loader


class Times(object):
    def __init__(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def start(self):
        self.st = time.time()

    def end(self, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += self.et - self.st
        else:
            self.time = self.et - self.st

    def reset(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def value(self):
        return round(self.time, 4)


def get_current_memory_mb(gpu_id=None):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024. / 1024.
    gpu_mem = 0
    gpu_precent = 0
    if gpu_id is not None:
        GPUs = GPUtil.getGPUs()
        gpu_load = GPUs[gpu_id].load
        gpu_precent = gpu_load
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024. / 1024.
    return cpu_mem, gpu_mem, gpu_precent


def main(args):
    predictor, pred_config = init_predictor(args)
    place = paddle.set_device('gpu' if args.use_gpu else 'cpu')
    args.place = place
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    test_dataloader = create_data_loader(args)
    preprocess_time = Times()
    inference_time = Times()
    postprocess_time = Times()
    cpu_mem, gpu_mem = 0, 0
    gpu_id = 0
    gpu_util = 0
    for batch_id, batch_data in enumerate(test_dataloader):
        name_data_pair = dict(zip(input_names, batch_data))
        preprocess_time.start()
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(name_data_pair[name].numpy())
        preprocess_time.end(accumulative=True)
        inference_time.start()
        predictor.run()
        for name in output_names:
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
        inference_time.end(accumulative=True)
        results = []
        results_type = []
        postprocess_time.start()
        for name in output_names:
            results_type.append(output_tensor.type())
            results.append(output_data[0])
        postprocess_time.end(accumulative=True)
        cm, gm, gu = get_current_memory_mb(gpu_id)
        cpu_mem += cm
        gpu_mem += gm
        gpu_util += gu
        print(results)

    num_test_data = args.batchsize * (batch_id + 1)
    average_preprocess_time = preprocess_time.value() / (batch_id + 1)
    average_inference_time = inference_time.value() / (batch_id + 1)
    average_postprocess_time = postprocess_time.value() / (batch_id + 1)
    cpu_rss = cpu_mem / (batch_id + 1)
    gpu_rss = gpu_mem / (batch_id + 1)
    gpu_util = gpu_util / (batch_id + 1)

    perf_info = {
        'inference_time_s': average_inference_time,
        'preprocess_time_s': average_preprocess_time,
        'postprocess_time_s': average_postprocess_time
    }
    model_info = {'model_name': args.model_name, 'precision': "fp32"}
    data_info = {
        'batch_size': args.batchsize,
        'shape': "dynamic_shape",
        'data_num': num_test_data
    }
    resource_info = {
        'cpu_rss_mb': cpu_rss,
        'gpu_rss_mb': gpu_rss,
        'gpu_util': gpu_util
    }
    rec_log = PaddleInferBenchmark(pred_config, model_info, data_info,
                                   perf_info, resource_info)
    rec_log('Rec')


if __name__ == '__main__':
    args = parse_args()
    main(args)
