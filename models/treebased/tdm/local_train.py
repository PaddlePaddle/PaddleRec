# -*- coding=utf-8 -*-
"""
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""

from __future__ import print_function
import os
import time
import numpy as np
import logging
import argparse
import multiprocessing

import paddle
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker

from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

from args import print_arguments, parse_args
from utils import UtilsWrapper, tdm_sampler_prepare, tdm_emb_prepare, load_tree_info, tdm_child_prepare
from train_net import TDM
#paddle.enable_static()

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def get_dataset(inputs, args):
    """
    get dataset
    """
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_pipe_command("python ./dataset_generator.py {}".format(
        args.bidid_leafid_path))
    dataset.set_batch_size(args.batch_size)
    dataset.set_thread(int(args.dataset_threads_num))
    file_list = [
        str(args.train_files_path) + "/%s" % x
        for x in os.listdir(args.train_files_path)
    ]
    dataset.set_filelist(file_list)
    logger.info("file list: {}".format(file_list))
    total_num = get_example_num(file_list)
    return dataset


def get_device_info(args):
    """
    get device info
    """
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    return place, dev_count


def run_train(args):
    """
    run train
    """
    # set random
    program = fluid.default_main_program()
    program.random_seed = args.random_seed

    # model
    logger.info("TDM Begin build network.")
    tdm_model = TDM(args)
    inputs = tdm_model.input_data()

    avg_cost, acc = tdm_model.create_model(inputs)
    logger.info("TDM End build network.")

    dataset = get_dataset(inputs, args)

    # optimizer
    if args.optimizer == "sgd":
        optimizer = fluid.optimizer.SGD(learning_rate=args.learning_rate)
    elif args.optimizer == "adam_lazy":
        optimizer = fluid.optimizer.AdamOptimizer(
            learning_rate=args.learning_rate,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon,
            lazy_mode=True)
    elif args.optimizer == "adam":
        optimizer = fluid.optimizer.AdamOptimizer(
            learning_rate=args.learning_rate,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon)
    else:
        optimizer = fluid.optimizer.AdamOptimizer(
            learning_rate=args.learning_rate,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon,
            lazy_mode=True)
    optimizer.minimize(avg_cost)
    logger.info("TDM End append backward.")

    # executor
    place, dev_count = get_device_info(args)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load params
    with open("main_program", 'w') as f:
        f.write(str(fluid.default_main_program()))

    if args.load_params:
        model_path = (str(args.output_model_files_path) + "/" + "init_model")
        fluid.io.load_persistables(exe, dirname=model_path)

        param_init = {}
        param_init['relu.w'] = np.load(
            "./thirdparty/init_param/relu_w_tensor_np.npy")
        param_init['relu.b'] = np.load(
            "./thirdparty/init_param/relu_b_tensor_np.npy")
        param_init['cos_sim.w'] = np.load(
            "./thirdparty/init_param/cos_sim_w_tensor_np.npy")
        param_init['cos_sim.b'] = np.load(
            "./thirdparty/init_param/cos_sim_b_tensor_np.npy")

        for name in param_init:
            print(name, param_init[name].shape)
            var = fluid.global_scope().find_var(name).get_tensor()
            print(np.array(var))
            var.set(param_init[name].astype('float32'), place)

        param_init['diag'] = np.diag(np.ones(args.y_index_embed_size))
        param_init['zero'] = np.zeros(args.y_index_embed_size)

        print(tdm_model.max_layers)
        for i in range(tdm_model.max_layers):
            var_w_name = "mt.q_layer_fc.weight.{}".format(str(i))
            var_b_name = "mt.q_layer_fc.bias.{}".format(str(i))
            print(var_w_name, var_b_name)
            w_var = fluid.global_scope().find_var(var_w_name).get_tensor()
            w_var.set(param_init['diag'].astype('float32'), place)
            b_var = fluid.global_scope().find_var(var_b_name).get_tensor()
            b_var.set(param_init['zero'].astype('float32'), place)

        lr = fluid.global_scope().find_var("learning_rate_0").get_tensor()
        lr.set(np.array(args.learning_rate).astype('float32'), place)
        logger.info("Load persistables finished.")

    logger.info("TDM Begin load tree travel & layer.")
    travel_list, travel_array, layer_list, layer_array = tdm_sampler_prepare(
        args)
    print(travel_array.shape)
    print(layer_array.shape)

    emb_array = tdm_child_prepare(args)
    print(emb_array.shape)
    Pytorch_model = {}
    Pytorch_model['TDM_Tree_Travel'] = travel_array
    Pytorch_model['TDM_Tree_Layer'] = layer_array
    Pytorch_model['TDM_Tree_Emb'] = emb_array
    logger.info("end")

    logger.info("TDM Begin load parameter.")
    param_name_list = ['TDM_Tree_Travel', 'TDM_Tree_Layer', 'TDM_Tree_Emb']
    for param_name in param_name_list:
        param_t = fluid.global_scope().find_var(param_name).get_tensor()
        param_t.set(Pytorch_model[str(param_name)].astype('int32'), place)
    # load tree embedding npy
    tdm_emb_prepare(args, place)

    if args.save_init_model:
        logger.info("Begin Save Init model.")
        model_path = (str(args.output_model_files_path) + "/" + "init_model")
        fluid.io.save_persistables(executor=exe, dirname=model_path)
        logger.info("End Save Init model.")

    # trace params
    #UtilsWrapper.trace_params(param_names, args.trace_params)
    """local train"""
    logger.info("TDM Local training begin ...")
    for epoch in range(args.epoch_num):
        start_time = time.time()
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=dataset,
            fetch_list=[acc, avg_cost],
            fetch_info=[
                "Epoch {} acc".format(epoch), "Epoch {} loss".format(epoch)
            ],
            print_period=10,
            debug=False, )
        end_time = time.time()
        logger.info("Epoch %d finished, use time=%d sec\n" %
                    (epoch, end_time - start_time))

        model_path = (str(args.output_model_files_path) + "/" +
                      str(args.job_name) + "_epoch_" + str(epoch))
        fluid.io.save_persistables(executor=exe, dirname=model_path)

    logger.info("Local training success!")


def get_example_num(file_list):
    """
    Count the number of samples in the file
    """
    count = 0
    for f in file_list:
        last_count = count
        for index, line in enumerate(open(f, 'r')):
            count += 1
        logger.info("file : %s has %s example" % (f, count - last_count))
    logger.info("Total example : %s" % count)
    return count


def main(args):
    """main"""
    np.random.seed(args.random_seed)

    if args.exec_mode == "train":
        run_train(args)
    else:
        logger.info("exec_mode supposed to be train")


if __name__ == "__main__":
    args = parse_args()
    load_tree_info(args)
    print_arguments(args)
    main(args)
