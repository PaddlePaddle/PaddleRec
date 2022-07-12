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

import paddle
import os
import time
import logging
import sys
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
print(os.path.abspath('/'.join(__dir__.split('/')[:-3])))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath('/'.join(__dir__.split('/')[:-3])))

from tools.utils.utils_single import load_yaml, load_dy_model_class, get_abs_model, create_data_loader
from tools.utils.save_load import save_model, load_model
from paddle.io import DataLoader
import argparse

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--config_yaml", type=str)
    parser.add_argument("-o", "--opt", nargs='*', type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def main(args):
    paddle.seed(2021)
    # load config
    config = load_yaml(args.config_yaml)
    dy_model_class = load_dy_model_class(args.abs_dir)
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

    # tools.vars
    use_gpu = config.get("runner.use_gpu", True)
    use_xpu = config.get("runner.use_xpu", False)
    use_npu = config.get("runner.use_npu", False)
    use_visual = config.get("runner.use_visual", False)
    test_data_dir = config.get("runner.test_data_dir", None)
    print_interval = config.get("runner.print_interval", None)
    infer_batch_size = config.get("runner.infer_batch_size", None)
    model_load_path = config.get("runner.infer_load_path", "model_output")
    start_epoch = config.get("runner.infer_start_epoch", 0)
    end_epoch = config.get("runner.infer_end_epoch", 10)
    infer_train_epoch = config.get("runner.infer_train_epoch", 2)
    batchsize = config.get("hyper_parameters.batch_size", 32)

    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, use_xpu: {}, use_npu: {}, use_visual: {}, infer_batch_size: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, use_xpu, use_npu, use_visual, infer_batch_size,
               test_data_dir, start_epoch, end_epoch, print_interval,
               model_load_path))
    logger.info("**************common.configs**********")

    if use_xpu:
        xpu_device = 'xpu:{0}'.format(os.getenv('FLAGS_selected_xpus', 0))
        place = paddle.set_device(xpu_device)
    elif use_npu:
        npu_device = 'npu:{0}'.format(os.getenv('FLAGS_selected_npus', 0))
        place = paddle.set_device(npu_device)
    else:
        place = paddle.set_device('gpu' if use_gpu else 'cpu')

    dy_model = dy_model_class.create_model(config)

    # Create a log_visual object and store the data in the path
    if use_visual:
        from visualdl import LogWriter
        log_visual = LogWriter(args.abs_dir + "/visualDL_log/infer")

    # to do : add optimizer function
    #optimizer = dy_model_class.create_optimizer(dy_model, config)

    logger.info("read data")
    infer_dataloader = create_data_loader(
        config=config, place=place, mode="test")

    epoch_begin = time.time()
    interval_begin = time.time()

    metric_list, metric_list_name = dy_model_class.create_metrics()
    step_num = 0
    print_interval = 1

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))

        infer_reader_cost = 0.0
        infer_run_cost = 0.0
        reader_start = time.time()

        assert any(infer_dataloader(
        )), "test_dataloader is null, please ensure batch size < dataset size!"

        aid_flag = -1

        for batch_id, batch in enumerate(infer_dataloader()):
            infer_reader_cost += time.time() - reader_start
            infer_start = time.time()

            aid_flag = batch[0][0].item()
            x_spt, y_spt, x_qry, y_qry = batch[1], batch[2], batch[3], batch[4]

            load_model(model_path, dy_model)
            # 对每个子任务进行训练
            optimizer = dy_model_class.create_optimizer(dy_model, config,
                                                        "infer")
            dy_model.train()

            for i in range(infer_train_epoch):
                n_samples = y_spt.shape[0]
                n_batch = int(np.ceil(n_samples / batchsize))
                optimizer.clear_grad()

                for i_batch in range(n_batch):
                    batch_input = list()
                    batch_x = []
                    batch_x.append(x_spt[0][i_batch * batchsize:(i_batch + 1) *
                                            batchsize])
                    batch_x.append(x_spt[1][i_batch * batchsize:(i_batch + 1) *
                                            batchsize])
                    batch_x.append(x_spt[2][i_batch * batchsize:(i_batch + 1) *
                                            batchsize])
                    batch_x.append(x_spt[3][i_batch * batchsize:(i_batch + 1) *
                                            batchsize])

                    batch_y = y_spt[i_batch * batchsize:(i_batch + 1) *
                                    batchsize]

                    batch_input.append(batch_x)
                    batch_input.append(batch_y)

                    loss = dy_model_class.infer_train_forward(
                        dy_model, batch_input, config)

                    dy_model.clear_gradients()
                    loss.backward()
                    optimizer.step()
            # 对每个子任务进行测试
            dy_model.eval()
            metric_list_local, metric_list_local_name = dy_model_class.create_metrics(
            )
            with paddle.no_grad():
                n_samples = y_qry.shape[0]
                n_batch = int(np.ceil(n_samples / batchsize))

                for i_batch in range(n_batch):
                    batch_input = list()
                    batch_x = []
                    batch_x.append(x_qry[0][i_batch * batchsize:(i_batch + 1) *
                                            batchsize])
                    batch_x.append(x_qry[1][i_batch * batchsize:(i_batch + 1) *
                                            batchsize])
                    batch_x.append(x_qry[2][i_batch * batchsize:(i_batch + 1) *
                                            batchsize])
                    batch_x.append(x_qry[3][i_batch * batchsize:(i_batch + 1) *
                                            batchsize])

                    batch_y = y_qry[i_batch * batchsize:(i_batch + 1) *
                                    batchsize]

                    batch_input.append(batch_x)
                    batch_input.append(batch_y)

                    metric_list, metric_list_local = dy_model_class.infer_forward(
                        dy_model, metric_list, metric_list_local, batch_input,
                        config)

            infer_run_cost += time.time() - infer_start

            metric_str_local = ""
            for metric_id in range(len(metric_list_local_name)):
                metric_str_local += (
                    metric_list_local_name[metric_id] + ": {:.6f},".format(
                        metric_list_local[metric_id].accumulate()))
                if use_visual:
                    log_visual.add_scalar(
                        tag="infer/" + metric_list_local_name[metric_id],
                        step=step_num,
                        value=metric_list_local[metric_id].accumulate())
            logger.info(
                "epoch: {}, batch_id: {}, aid: {} ".format(
                    epoch_id, batch_id, aid_flag) + metric_str_local +
                " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.2f} ins/s".
                format(infer_reader_cost / print_interval, (
                    infer_reader_cost + infer_run_cost) / print_interval,
                       batchsize, print_interval * batchsize / (time.time(
                       ) - interval_begin)))

            interval_begin = time.time()
            infer_reader_cost = 0.0
            infer_run_cost = 0.0
            step_num = step_num + 1
            reader_start = time.time()

        metric_str = ""
        for metric_id in range(len(metric_list_name)):
            metric_str += (
                metric_list_name[metric_id] +
                ": {:.6f},".format(metric_list[metric_id].accumulate()))

        logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                    " epoch time: {:.2f} s".format(time.time() - epoch_begin))
        epoch_begin = time.time()


if __name__ == '__main__':
    args = parse_args()
    main(args)
