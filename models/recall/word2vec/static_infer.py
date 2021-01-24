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

from __future__ import print_function
import os
import warnings
import logging
import paddle
import sys
import numpy as np
__dir__ = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from utils.utils_single import load_yaml, load_static_model_class, get_abs_model, create_data_loader, reset_auc
from utils.save_load import save_static_model, load_static_model

import time
import argparse

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("PaddleRec train static script")
    parser.add_argument("-m", "--config_yaml", type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def main(args):
    paddle.seed(12345)

    # load config
    config = load_yaml(args.config_yaml)
    config["config_abs_dir"] = args.abs_dir
    # load static model class
    static_model_class = load_static_model_class(config)

    input_data = static_model_class.create_feeds(is_infer=True)
    input_data_names = [data.name for data in input_data]

    fetch_vars = static_model_class.infer_net(input_data)
    logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))

    use_gpu = config.get("runner.use_gpu", True)
    use_auc = config.get("runner.use_auc", False)
    test_data_dir = config.get("runner.test_data_dir", None)
    print_interval = config.get("runner.print_interval", None)
    model_load_path = config.get("runner.infer_load_path", "model_output")
    start_epoch = config.get("runner.infer_start_epoch", 0)
    end_epoch = config.get("runner.infer_end_epoch", 10)
    batch_size = config.get("runner.infer_batch_size", None)
    sparse_feature_number = config.get(
        "hyper_parameters.sparse_feature_number")
    os.environ["CPU_NUM"] = str(config.get("runner.thread_num", 1))
    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, test_data_dir, start_epoch, end_epoch, print_interval,
               model_load_path))
    logger.info("**************common.configs**********")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')
    exe = paddle.static.Executor(place)
    # initialize
    exe.run(paddle.static.default_startup_program())

    test_dataloader = create_data_loader(
        config=config, place=place, mode="test")

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_static_model(
            paddle.static.default_main_program(),
            model_path,
            prefix='rec_static')

        accum_num_sum = 0
        accum_num = 0
        epoch_begin = time.time()
        interval_begin = time.time()
        for batch_id, batch_data in enumerate(test_dataloader()):
            #print(np.array(batch_data[0]))
            ##b_size = len([dat[0] for dat in batch_data])
            #print(b_size)
            #wa = np.array([dat[0] for dat in batch_data]).astype(
            #            "int64").reshape(b_size)
            #wb = np.array([dat[1] for dat in batch_data]).astype(
            #            "int64").reshape(b_size)
            #wc = np.array([dat[2] for dat in batch_data]).astype(
            #            "int64").reshape(b_size)
            fetch_batch_var = exe.run(
                program=paddle.static.default_main_program(),
                feed={
                    "analogy_a": np.array(batch_data[0]),
                    "analogy_b": np.array(batch_data[1]),
                    "analogy_c": np.array(batch_data[2]),
                    "all_label": np.arange(sparse_feature_number)
                    .reshape(sparse_feature_number).astype("int64")
                },
                fetch_list=[var for _, var in fetch_vars.items()])
            pre = np.array(fetch_batch_var[0])
            #pre = pred_idx.numpy()
            label = np.array(batch_data[3])
            inputs_word = np.array(batch_data[4])

            for ii in range(len(label)):
                top4 = pre[ii][0]
                accum_num_sum += 1
                for idx in top4:
                    if int(idx) in inputs_word[ii]:
                        continue
                    if int(idx) == int(label[ii][0]):
                        accum_num += 1
                    break

            if batch_id % print_interval == 0:
                logger.info(
                    "infer epoch: {}, batch_id: {}, acc: {:.6f}, speed: {:.2f} ins/s".
                    format(epoch_id, batch_id, accum_num * 1.0 / accum_num_sum,
                           print_interval * batch_size / (time.time() -
                                                          interval_begin)))
                interval_begin = time.time()
        logger.info("infer epoch: {} done, acc: {:.6f}, : epoch time{:.2f} s".
                    format(epoch_id, accum_num * 1.0 / accum_num_sum,
                           time.time() - epoch_begin))

        epoch_begin = time.time()


if __name__ == "__main__":
    paddle.enable_static()
    args = parse_args()
    main(args)
