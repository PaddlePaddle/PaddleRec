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

from utils.static_ps.reader_helper import get_reader
from utils.utils_single import load_yaml, load_static_model_class, get_abs_model, create_data_loader, reset_auc
from utils.save_load import save_static_model, save_inference_model, load_static_parameter, save_data

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
    config["yaml_path"] = args.config_yaml
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
    input_data = static_model_class.create_feeds()
    input_data_names = [data.name for data in input_data]

    fetch_vars = static_model_class.net(input_data)

    #infer_target_var = model.infer_target_var
    logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))

    use_gpu = config.get("runner.use_gpu", True)
    use_xpu = config.get("runner.use_xpu", False)
    use_auc = config.get("runner.use_auc", False)
    use_visual = config.get("runner.use_visual", False)
    use_inference = config.get("runner.use_inference", False)
    auc_num = config.get("runner.auc_num", 1)
    train_data_dir = config.get("runner.train_data_dir", None)
    epochs = config.get("runner.epochs", None)
    print_interval = config.get("runner.print_interval", None)
    model_save_path = config.get("runner.model_save_path", "model_output")
    model_init_path = config.get("runner.model_init_path", None)
    batch_size = config.get("runner.train_batch_size", None)
    reader_type = config.get("runner.reader_type", "DataLoader")
    use_fleet = config.get("runner.use_fleet", False)
    use_save_data = config.get("runner.use_save_data", False)
    os.environ["CPU_NUM"] = str(config.get("runner.thread_num", 1))
    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, use_xpu: {}, use_visual: {}, train_batch_size: {}, train_data_dir: {}, epochs: {}, print_interval: {}, model_save_path: {}".
        format(use_gpu, use_xpu, use_visual, batch_size, train_data_dir,
               epochs, print_interval, model_save_path))
    logger.info("**************common.configs**********")

    if use_xpu:
        xpu_device = 'xpu:{0}'.format(os.getenv('FLAGS_selected_xpus', 0))
        place = paddle.set_device(xpu_device)
    else:
        place = paddle.set_device('gpu' if use_gpu else 'cpu')

    if use_fleet:
        from paddle.distributed import fleet
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)
    if use_fleet:
        static_model_class.create_optimizer(strategy)
    else:
        static_model_class.create_optimizer()

    exe = paddle.static.Executor(place)
    # initialize
    exe.run(paddle.static.default_startup_program())

    if model_init_path is not None:
        load_static_parameter(
            paddle.static.default_main_program(),
            model_init_path,
            prefix='rec_static')

    last_epoch_id = config.get("last_epoch", -1)

    # Create a log_visual object and store the data in the path
    if use_visual:
        from visualdl import LogWriter
        log_visual = LogWriter(args.abs_dir + "/visualDL_log/train")
    else:
        log_visual = None
    step_num = 0

    if reader_type == 'QueueDataset':
        dataset, file_list = get_reader(input_data, config)
    elif reader_type == 'DataLoader':
        train_dataloader = create_data_loader(config=config, place=place)
    elif reader_type == "CustomizeDataLoader":
        train_dataloader = static_model_class.create_data_loader()
        reader_type = 'DataLoader'

    for epoch_id in range(last_epoch_id + 1, epochs):

        epoch_begin = time.time()
        if use_auc:
            reset_auc(use_fleet, auc_num)
        if reader_type == 'DataLoader':
            fetch_batch_var, step_num = dataloader_train(
                epoch_id, train_dataloader, input_data_names, fetch_vars, exe,
                config, use_visual, log_visual, step_num)
            metric_str = ""
            for var_idx, var_name in enumerate(fetch_vars):
                metric_str += "{}: {}, ".format(
                    var_name, str(fetch_batch_var[var_idx]).strip("[]"))
            logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                        "epoch time: {:.2f} s".format(time.time() -
                                                      epoch_begin))
        elif reader_type == 'QueueDataset':
            fetch_batch_var = dataset_train(epoch_id, dataset, fetch_vars, exe,
                                            config)
            logger.info("epoch: {} done, ".format(epoch_id) +
                        "epoch time: {:.2f} s".format(time.time() -
                                                      epoch_begin))
        else:
            logger.info("reader type wrong")

        if use_fleet:
            trainer_id = paddle.distributed.get_rank()
            if trainer_id == 0:
                save_static_model(
                    paddle.static.default_main_program(),
                    model_save_path,
                    epoch_id,
                    prefix='rec_static')
        else:
            save_static_model(
                paddle.static.default_main_program(),
                model_save_path,
                epoch_id,
                prefix='rec_static')
        if use_save_data:
            save_data(fetch_batch_var, model_save_path)

        if use_inference:
            feed_var_names = config.get("runner.save_inference_feed_varnames",
                                        [])
            feedvars = []
            fetch_var_names = config.get(
                "runner.save_inference_fetch_varnames", [])
            fetchvars = []
            for var_name in feed_var_names:
                if var_name not in paddle.static.default_main_program(
                ).global_block().vars:
                    raise ValueError(
                        "Feed variable: {} not in default_main_program, global block has follow vars: {}".
                        format(var_name,
                               paddle.static.default_main_program()
                               .global_block().vars.keys()))
                else:
                    feedvars.append(paddle.static.default_main_program()
                                    .global_block().vars[var_name])
            for var_name in fetch_var_names:
                if var_name not in paddle.static.default_main_program(
                ).global_block().vars:
                    raise ValueError(
                        "Fetch variable: {} not in default_main_program, global block has follow vars: {}".
                        format(var_name,
                               paddle.static.default_main_program()
                               .global_block().vars.keys()))
                else:
                    fetchvars.append(paddle.static.default_main_program()
                                     .global_block().vars[var_name])

            save_inference_model(model_save_path, epoch_id, feedvars,
                                 fetchvars, exe)


def dataset_train(epoch_id, dataset, fetch_vars, exe, config):
    #logger.info("Epoch: {}, Running Dataset Begin.".format(epoch))
    fetch_info = [
        "Epoch {} Var {}".format(epoch_id, var_name) for var_name in fetch_vars
    ]
    fetch_vars = [var for _, var in fetch_vars.items()]
    print_interval = config.get("runner.print_interval")
    exe.train_from_dataset(
        program=paddle.static.default_main_program(),
        dataset=dataset,
        fetch_list=fetch_vars,
        fetch_info=fetch_info,
        print_period=print_interval,
        debug=config.get("runner.dataset_debug"))


def dataloader_train(epoch_id, train_dataloader, input_data_names, fetch_vars,
                     exe, config, use_visual, log_visual, step_num):
    print_interval = config.get("runner.print_interval", None)
    batch_size = config.get("runner.train_batch_size", None)
    interval_begin = time.time()
    train_reader_cost = 0.0
    train_run_cost = 0.0
    total_samples = 0
    reader_start = time.time()
    for batch_id, batch_data in enumerate(train_dataloader()):
        train_reader_cost += time.time() - reader_start
        train_start = time.time()

        fetch_batch_var = exe.run(
            program=paddle.static.default_main_program(),
            feed=dict(zip(input_data_names, batch_data)),
            fetch_list=[var for _, var in fetch_vars.items()])

        train_run_cost += time.time() - train_start
        total_samples += batch_size
        if batch_id % print_interval == 0:
            metric_str = ""
            for var_idx, var_name in enumerate(fetch_vars):
                metric_str += "{}: {}, ".format(
                    var_name, str(fetch_batch_var[var_idx]).strip("[]"))
                if use_visual:
                    log_visual.add_scalar(
                        tag="train/" + var_name,
                        step=step_num,
                        value=fetch_batch_var[var_idx])
            logger.info(
                "epoch: {}, batch_id: {}, ".format(epoch_id,
                                                   batch_id) + metric_str +
                "avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} ins/s".
                format(train_reader_cost / print_interval, (
                    train_reader_cost + train_run_cost) / print_interval,
                       total_samples / print_interval, total_samples / (
                           train_reader_cost + train_run_cost)))
            train_reader_cost = 0.0
            train_run_cost = 0.0
            total_samples = 0
        reader_start = time.time()
        step_num = step_num + 1
    return fetch_batch_var, step_num


if __name__ == "__main__":
    paddle.enable_static()
    args = parse_args()
    main(args)
