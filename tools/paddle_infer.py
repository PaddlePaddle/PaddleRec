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
    parser.add_argument("--model_name", type=str, default="not specified")
    args = parser.parse_args()
    args.use_gpu = (True if args.use_gpu.lower() == "true" else False)
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
        print(config)
        # config.delete('repeated_fc_relu_fuse_pass')
    predictor = create_predictor(config)
    return predictor


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


def log_print(args, results_type, num_test_data, average_preprocess_time,
              average_inference_time, average_postprocess_time, cpu_rss,
              gpu_rss, gpu_util):
    print("----------------------- Model info ----------------------")
    print("model_name: {}\ntype: {}\nmodel_sorce: {}".format(
        args.model_name, "static", "PaddleRec"))
    print("----------------------- Data info -----------------------")
    print("batch_size: {}".format(args.batchsize))
    print("----------------------- Conf info -----------------------")
    print("runtime_device: {}".format("gpu" if args.use_gpu else "cpu"))
    print("ir_optim: {}\nenable_memory_optim: {}\nenable_tensorrt: {}".format(
        "False", "False", "False"))
    print("precision: {}".format([str(x).split(".")[1] for x in results_type]))
    print("enable_mkldnn: {}\ncpu_math_library_num_threads: {}".format("False",
                                                                       1))
    print("----------------------- Perf info -----------------------")
    print(
        "preprocess_time(ms): {}\ninference_time(ms): {}\npostprocess_time(ms): {}".
        format(average_preprocess_time * 1000, average_inference_time * 1000,
               average_postprocess_time * 1000))
    print("The number of predicted data: {}".format(num_test_data))
    print("cpu_rss(MB): {}, gpu_rss(MB): {}".format(cpu_rss, gpu_rss))
    print("gpu_util: {}%".format(str(gpu_util * 100)[:4]))


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
    predictor = init_predictor(args)
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
        inference_time.end(accumulative=True)
        results = []
        results_type = []
        postprocess_time.start()
        for name in output_names:
            output_tensor = predictor.get_output_handle(name)
            results_type.append(output_tensor.type())
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data[0])
        postprocess_time.end(accumulative=True)
        cm, gm, gu = get_current_memory_mb(gpu_id)
        cpu_mem += cm
        gpu_mem += gm
        gpu_util += gu
        print(results)

    num_test_data = args.batchsize * (batch_id + 1)
    average_preprocess_time = preprocess_time.value() / num_test_data
    average_inference_time = inference_time.value() / num_test_data
    average_postprocess_time = postprocess_time.value() / num_test_data
    cpu_rss = cpu_mem / num_test_data
    gpu_rss = gpu_mem / num_test_data
    gpu_util = gpu_util / num_test_data
    log_print(args, results_type, num_test_data, average_preprocess_time,
              average_inference_time, average_postprocess_time, cpu_rss,
              gpu_rss, gpu_util)


if __name__ == '__main__':
    args = parse_args()
    main(args)
