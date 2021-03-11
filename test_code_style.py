#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import argparse
import time
import numpy as np
from paddle.inference import Config, PrecisionType
from paddle.inference import create_predictor


def main():
    args = parse_args()
    config = set_config(args)
    predictor = create_predictor(config)
    input_names = predictor.get_input_names()
    input_hanlde = predictor.get_input_handle(input_names[0])
    fake_input = np.ones((args.batch_size, 3, 224, 224)).astype("float32")
    input_hanlde.reshape([args.batch_size, 3, 224, 224])
    input_hanlde.copy_from_cpu(fake_input)
    for i in range(args.warmup):
        predictor.run()
    start_time = time.time()
    for i in range(args.repeats):
        predictor.run()
    output_names =predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()
    end_time = time.time()
    print(output_data[0, :10])
    print('time is: {}'.format((end_time - start_time) / args.repeats * 1000))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,help="model dir")
    parser.add_argument("--model_file", type=str,help="model filename")
    parser.add_argument("--params_file", type=str, help="parameter filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--warmup", type=int, default=0, help="warmup")
    parser.add_argument("--repeats", type=int, default=1, help="repeats")
    parser.add_argument("--math_thread_num", type=int, default=1, help="math_thread_num")
    return parser.parse_args()


def set_config(args):
    config = Config(args.model_file, args.params_file)
    config.enable_lite_engine(PrecisionType.Float32,
                              True)
    # use lite xpu subgraph
    config.enable_xpu(10 * 1024 * 1024)
    # use lite cuda subgraph
    # config.enable_use_gpu(100, 0)
    config.set_cpu_math_library_num_threads(args.math_thread_num)
    return config


if __name__ == "__main__":
    main()

