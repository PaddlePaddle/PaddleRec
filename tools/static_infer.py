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
__dir__ = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from utils.utils_single import load_yaml, load_static_model_class, get_abs_model, create_data_loader, reset_auc
from utils.save_load import save_static_model, load_static_model, save_data
import time
import argparse

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("PaddleRec train static script")
    parser.add_argument("-m", "--config_yaml", type=str)
    parser.add_argument("-o", "--opt", nargs='*', type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def main(args):
    paddle.seed(12345)

    # load config
    config = load_yaml(args.config_yaml)
    config["config_abs_dir"] = args.abs_dir
    # modify config from command
    if args.opt:
        for parameter in args.opt:
            parameter = parameter.strip()
            key, value = parameter.split("=")
            if type(config.get(key)) is int:
                value = int(value)
            if type(config.get(key)) is float:
                value = float(value)
            if type(config.get(key)) is bool:
                value = (True if value.lower() == "true" else False)
            config[key] = value
    # load static model class
    static_model_class = load_static_model_class(config)

    input_data = static_model_class.create_feeds(is_infer=True)
    input_data_names = [data.name for data in input_data]

    fetch_vars = static_model_class.infer_net(input_data)
    logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))

    use_gpu = config.get("runner.use_gpu", True)
    use_xpu = config.get("runner.use_xpu", False)
    use_auc = config.get("runner.use_auc", False)
    use_visual = config.get("runner.use_visual", False)
    auc_num = config.get("runner.auc_num", 1)
    test_data_dir = config.get("runner.test_data_dir", None)
    print_interval = config.get("runner.print_interval", None)
    model_load_path = config.get("runner.infer_load_path", "model_output")
    start_epoch = config.get("runner.infer_start_epoch", 0)
    end_epoch = config.get("runner.infer_end_epoch", 10)
    batch_size = config.get("runner.infer_batch_size", None)
    use_save_data = config.get("runner.use_save_data", False)
    reader_type = config.get("runner.reader_type", "DataLoader")
    use_fleet = config.get("runner.use_fleet", False)
    os.environ["CPU_NUM"] = str(config.get("runner.thread_num", 1))
    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, use_xpu: {}, use_visual: {}, infer_batch_size: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, use_xpu, use_visual, batch_size, test_data_dir,
               start_epoch, end_epoch, print_interval, model_load_path))
    logger.info("**************common.configs**********")

    if use_xpu:
        xpu_device = 'xpu:{0}'.format(os.getenv('FLAGS_selected_xpus', 0))
        place = paddle.set_device(xpu_device)
    else:
        place = paddle.set_device('gpu' if use_gpu else 'cpu')
    exe = paddle.static.Executor(place)
    # initialize
    exe.run(paddle.static.default_startup_program())

    if reader_type == 'DataLoader':
        test_dataloader = create_data_loader(
            config=config, place=place, mode="test")
    elif reader_type == "CustomizeDataLoader":
        test_dataloader = static_model_class.create_data_loader()

    # Create a log_visual object and store the data in the path
    if use_visual:
        from visualdl import LogWriter
        log_visual = LogWriter(args.abs_dir + "/visualDL_log/infer")
    step_num = 0

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_static_model(
            paddle.static.default_main_program(),
            model_path,
            prefix='rec_static')

        epoch_begin = time.time()
        interval_begin = time.time()
        infer_reader_cost = 0.0
        infer_run_cost = 0.0
        reader_start = time.time()

        if use_auc:
            reset_auc(use_fleet, auc_num)
        for batch_id, batch_data in enumerate(test_dataloader()):
            infer_reader_cost += time.time() - reader_start
            infer_start = time.time()
            fetch_batch_var = exe.run(
                program=paddle.static.default_main_program(),
                feed=dict(zip(input_data_names, batch_data)),
                fetch_list=[var for _, var in fetch_vars.items()])
            infer_run_cost += time.time() - infer_start
            if batch_id % print_interval == 0:
                metric_str = ""
                for var_idx, var_name in enumerate(fetch_vars):
                    metric_str += "{}: {}, ".format(
                        var_name, fetch_batch_var[var_idx][0])
                    if use_visual:
                        log_visual.add_scalar(
                            tag="infer/" + var_name,
                            step=step_num,
                            value=fetch_batch_var[var_idx][0])
                logger.info(
                    "epoch: {}, batch_id: {}, ".format(epoch_id,
                                                       batch_id) + metric_str +
                    "avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.2f} ins/s".
                    format(infer_reader_cost / print_interval, (
                        infer_reader_cost + infer_run_cost) / print_interval,
                           batch_size, print_interval * batch_size / (
                               time.time() - interval_begin)))
                interval_begin = time.time()
                infer_reader_cost = 0.0
                infer_run_cost = 0.0
            reader_start = time.time()
            step_num = step_num + 1

        metric_str = ""
        for var_idx, var_name in enumerate(fetch_vars):
            metric_str += "{}: {}, ".format(var_name,
                                            fetch_batch_var[var_idx][0])
        logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                    "epoch time: {:.2f} s".format(time.time() - epoch_begin))
        if use_save_data:
            save_data(fetch_batch_var, model_load_path)


if __name__ == "__main__":
    paddle.enable_static()
    args = parse_args()
    main(args)
