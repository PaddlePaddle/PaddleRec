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
from numpy import *
import time
import logging
import sys
import threading
from threading import Thread
from importlib import import_module
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
from utils.utils_single import load_yaml, load_dy_model_class, get_abs_model
from utils.save_load import save_model, load_model
from paddle.io import DistributedBatchSampler, DataLoader
import argparse
from paddle.inference import Config
from paddle.inference import create_predictor
#import pynvml
#import psutil
#import GPUtil
import multiprocessing
from queue import Queue
import paddle.distributed.fleet as fleet

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

q = Queue()


class Reader(fleet.MultiSlotDataGenerator):
    def init(self, args):
        self.args = args
        self.padding = 0
        self.slots = self.slot_reader()
        self.fea_dict = {}
        self.visit = {}
        for slot in self.slots:
            self.visit[slot] = False

    def slot_reader(self):
        slot_num = 300
        slots = []
        if slot_num > 0:
            for i in range(2, slot_num + 2):
                slots.append(str(i))
        else:
            with open("./slot", "r") as rf:
                for line in rf.readlines():
                    slots.append(line.strip())
        return slots

    def batch_process(self, lines):
        output = {}
        lod = {}
        for slot in self.slots:
            output[slot] = []
            lod[slot] = [[0]]

        for line in lines:
            line = line.strip().split(" ")
            for i in range(len(line)):
                if i == 0:
                    continue
                slot_feasign = line[i].split(":")
                if len(slot_feasign) < 2:
                    print(i)
                slot = slot_feasign[1]
                if slot not in self.slots:
                    continue
                feasign = int(slot_feasign[0])
                if feasign not in self.fea_dict:
                    self.fea_dict[feasign] = len(self.fea_dict)
                output[slot].append(self.fea_dict[feasign])
                self.visit[slot] = True
            for slot in self.visit:
                if not self.visit[slot]:
                    output[slot].extend([self.padding])
                else:
                    self.visit[slot] = False
            for slot in self.slots:
                lod[slot][0].append(len(output[slot]))

        return output, lod

    def generate_batchSample(self):
        with open(self.args.reader_file, "r") as rf:
            cnt = 0
            lines = []
            for line in rf.readlines():
                cnt += 1
                lines.append(line)
                if (cnt == args.batchsize):
                    output, lod = self.batch_process(lines)
                    cnt = 0
                    i = 0
                    while True:
                        if i >= args.iteration_num:
                            break
                        q.put([lod, output])
                        i += 1
        return q.qsize()


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
        config.switch_ir_optim()
        config.delete_pass("repeated_fc_relu_fuse_pass")
        config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            config.enable_mkldnn()
    predictor = create_predictor(config)
    return predictor, config


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


def infer_main(predictor_cost_time, i, args):
    tid = threading.currentThread()
    print('Thread id : %d' % tid.ident)
    predictor, pred_config = init_predictor(args)
    place = paddle.set_device('gpu' if args.use_gpu else 'cpu')
    args.place = place
    input_names = predictor.get_input_names()
    results = {}
    output_names = predictor.get_output_names()
    for output_name in output_names:
        results[output_name] = []

    global q
    fin = open(args.log_file, 'w')
    batchCnt = 0
    t = Times()
    while True:
        if q.empty():
            break
        blob = q.get()
        batchCnt += 1
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
            input_tensor.set_lod(blob[0][name])
            nd = np.array(blob[1][name]).reshape((len(blob[1][name]), 1))
            input_tensor.copy_from_cpu(nd)
            ''' 
            fin.write("lod: ")
            for i in blob[0][name]:
                fin.write(str(i) + " ")
            fin.write("data: ")
            for i in blob[1][name]:
                fin.write(str(i) + " ")
            fin.write("\n")
            '''
        if args.test_predictor:
            t.start()
        predictor.run()
        if args.test_predictor:
            t.end(accumulative=True)
        '''
        for name in output_names:
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results[name].append(output_data)
        '''
        #fin.write("thread: " + str(tid.ident) + ", ")
        #fin.write("result: " + "\n")
        #fin.write(str(output_data))
    logger.info("processed batch num: {}".format(batchCnt))
    logger.info("predictor time cost: {}".format(t.value()))
    if args.test_predictor:
        predictor_cost_time[i] = t.value()
    fin.close()
    return results


class WrapperThread(Thread):
    def __init__(self, func, cost_time, thread_id, args):
        super(WrapperThread, self).__init__()
        self.func = func
        self.cost_time = cost_time
        self.thread_id = thread_id
        self.args = args

    def run(self):
        self.result = self.func(self.cost_time, self.thread_id, self.args)

    def get_result(self):
        return self.result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_predictor",
        type=str,
        default="False",
        help="test paddle predictor")
    parser.add_argument("--thread_num", type=int, default=2, help="thread num")
    parser.add_argument("--batchsize", type=int, default=5, help="batch size")
    parser.add_argument(
        "--iteration_num", type=int, default=10, help="iteration num")
    parser.add_argument(
        "--reader_file", default="../data/out_test.1", type=str)
    parser.add_argument(
        "--params_file", default="../model/rec_inference.pdiparams", type=str)
    parser.add_argument(
        "--model_file", default="../model/rec_inference.pdmodel", type=str)
    parser.add_argument("--log_file", default="./slot_dnn_infer.log", type=str)
    parser.add_argument(
        "--performance_file", default="./performance.txt", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--use_gpu", type=str, default="False")
    parser.add_argument("--data_dir", type=str)
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
    args.test_predictor = (True
                           if args.test_predictor.lower() == "true" else False)
    return args


if __name__ == '__main__':
    args = parse_args()

    reader = Reader()
    reader.init(args)
    batchCnt = reader.generate_batchSample()
    samplesCnt = batchCnt * args.batchsize
    thread_num = args.thread_num
    predictor_cost_time = list(range(thread_num))
    threads = []
    ff = open(args.performance_file, 'a')
    inference_time = Times()
    inference_time.start()
    for i in range(thread_num):
        t = WrapperThread(infer_main, predictor_cost_time, i, args=args)
        threads.append(t)
        t.start()
    for i in threads:
        t.join()
    inference_time.end(accumulative=True)
    cost_time = inference_time.value()
    ff.write("thread num: " + str(args.thread_num) + "\n")
    ff.write("batch size: " + str(args.batchsize) + "\n")
    ff.write("total sample num: " + str(samplesCnt) + "\n")
    qps = samplesCnt / cost_time
    ff.write("qps: " + str(qps) + "\n")
    latency = 1 / qps
    ff.write("latency: " + str(latency) + "\n")
    if args.test_predictor:
        predictor_cost_time_mean = mean(predictor_cost_time)
        predictor_qps = samplesCnt / predictor_cost_time_mean
        ff.write("paddle predictor time cost: " + str(predictor_cost_time_mean)
                 + "\n")
        ff.write("paddle predictor qps: " + str(predictor_qps) + "\n")
    ff.write("++++++++++++++++++++++++++++++++\n")
