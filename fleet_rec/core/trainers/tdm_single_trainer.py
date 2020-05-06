# -*- coding=utf-8 -*-
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

"""
Training use fluid with one node only.
"""

from __future__ import print_function
import logging
import paddle.fluid as fluid

from fleetrec.core.trainers.transpiler_trainer import TranspileTrainer
from fleetrec.core.trainers.single_trainer import SingleTrainer
from fleetrec.core.utils import envs
import numpy as np

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)
special_param = ["TDM_Tree_Travel", "TDM_Tree_Layer",
                 "TDM_Tree_Info", "TDM_Tree_Emb"]


class TDMSingleTrainer(SingleTrainer):
    def processor_register(self):
        self.regist_context_processor('uninit', self.instance)
        self.regist_context_processor('init_pass', self.init)
        self.regist_context_processor('startup_pass', self.startup)

        if envs.get_platform() == "LINUX":
            self.regist_context_processor('train_pass', self.dataset_train)
        else:
            self.regist_context_processor('train_pass', self.dataloader_train)

        self.regist_context_processor('infer_pass', self.infer)
        self.regist_context_processor('terminal_pass', self.terminal)

    def init(self, context):
        self.model.train_net()
        optimizer = self.model.optimizer()
        optimizer.minimize((self.model.get_cost_op()))

        self.fetch_vars = []
        self.fetch_alias = []
        self.fetch_period = self.model.get_fetch_period()

        metrics = self.model.get_metrics()
        if metrics:
            self.fetch_vars = metrics.values()
            self.fetch_alias = metrics.keys()
        context['status'] = 'startup_pass'

    def startup(self, context):
        namespace = "train.startup"
        load_persistables = envs.get_global_env(
            "single.load_persistables", False, namespace)
        persistables_model_path = envs.get_global_env(
            "single.persistables_model_path", "", namespace)

        load_tree = envs.get_global_env(
            "single.load_tree", False, namespace)
        self.tree_layer_path = envs.get_global_env(
            "single.tree_layer_path", "", namespace)
        self.tree_travel_path = envs.get_global_env(
            "single.tree_travel_path", "", namespace)
        self.tree_info_path = envs.get_global_env(
            "single.tree_info_path", "", namespace)
        self.tree_emb_path = envs.get_global_env(
            "single.tree_emb_path", "", namespace)

        save_init_model = envs.get_global_env(
            "single.save_init_model", False, namespace)
        init_model_path = envs.get_global_env(
            "single.init_model_path", "", namespace)
        self._exe.run(fluid.default_startup_program())

        if load_persistables:
            # 从paddle二进制模型加载参数
            fluid.io.load_persistables(
                executor=self._exe,
                dirname=persistables_model_path,
                main_program=fluid.default_main_program())
            logger.info("Load persistables from \"{}\"".format(
                persistables_model_path))

        if load_tree:
            # 将明文树结构及数据，set到组网中的Variale中
            # 不使用NumpyInitialize方法是考虑到树结构相关数据size过大，有性能风险
            for param_name in special_param:
                param_t = fluid.global_scope().find_var(param_name).get_tensor()
                param_array = self.tdm_prepare(param_name)
                if param_name == 'TDM_Tree_Emb':
                    param_t.set(param_array.astype('float32'), self._place)
                else:
                    param_t.set(param_array.astype('int32'), self._place)

        if save_init_model:
            logger.info("Begin Save Init model.")
            fluid.io.save_persistables(
                executor=self._exe, dirname=init_model_path)
            logger.info("End Save Init model.")

        context['status'] = 'train_pass'

    def dataloader_train(self, context):
        reader = self._get_dataloader()
        epochs = envs.get_global_env("train.epochs")

        program = fluid.compiler.CompiledProgram(
            fluid.default_main_program()).with_data_parallel(
            loss_name=self.model.get_cost_op().name)

        metrics_varnames = []
        metrics_format = []

        metrics_format.append("{}: {{}}".format("epoch"))
        metrics_format.append("{}: {{}}".format("batch"))

        for name, var in self.model.get_metrics().items():
            metrics_varnames.append(var.name)
            metrics_format.append("{}: {{}}".format(name))

        metrics_format = ", ".join(metrics_format)

        for epoch in range(epochs):
            reader.start()
            batch_id = 0
            try:
                while True:
                    metrics_rets = self._exe.run(
                        program=program,
                        fetch_list=metrics_varnames)

                    metrics = [epoch, batch_id]
                    metrics.extend(metrics_rets)

                    if batch_id % 10 == 0 and batch_id != 0:
                        print(metrics_format.format(*metrics))
                    batch_id += 1
            except fluid.core.EOFException:
                reader.reset()

        context['status'] = 'infer_pass'

    def dataset_train(self, context):
        dataset = self._get_dataset()
        epochs = envs.get_global_env("train.epochs")

        for i in range(epochs):
            self._exe.train_from_dataset(program=fluid.default_main_program(),
                                         dataset=dataset,
                                         fetch_list=self.fetch_vars,
                                         fetch_info=self.fetch_alias,
                                         print_period=self.fetch_period)
            self.save(i, "train", is_fleet=False)
        context['status'] = 'infer_pass'

    def infer(self, context):
        context['status'] = 'terminal_pass'

    def terminal(self, context):
        for model in self.increment_models:
            print("epoch :{}, dir: {}".format(model[0], model[1]))
        context['is_exit'] = True

    def tdm_prepare(self, param_name):
        if param_name == "TDM_Tree_Travel":
            travel_array = self.tdm_travel_prepare()
            return travel_array
        elif param_name == "TDM_Tree_Layer":
            layer_array, _ = self.tdm_layer_prepare()
            return layer_array
        elif param_name == "TDM_Tree_Info":
            info_array = self.tdm_info_prepare()
            return info_array
        elif param_name == "TDM_Tree_Emb":
            emb_array = self.tdm_emb_prepare()
            return emb_array
        else:
            raise " {} is not a special tdm param name".format(param_name)

    def tdm_travel_prepare(self):
        """load tdm tree param from npy/list file"""
        travel_array = np.load(self.tree_travel_path)
        logger.info("TDM Tree leaf node nums: {}".format(
            travel_array.shape[0]))
        return travel_array

    def tdm_emb_prepare(self):
        """load tdm tree param from npy/list file"""
        emb_array = np.load(self.tree_emb_path)
        logger.info("TDM Tree node nums from emb: {}".format(
            emb_array.shape[0]))
        return emb_array

    def tdm_layer_prepare(self):
        """load tdm tree param from npy/list file"""
        layer_list = []
        layer_list_flat = []
        with open(self.tree_layer_path, 'r') as fin:
            for line in fin.readlines():
                l = []
                layer = (line.split('\n'))[0].split(',')
                for node in layer:
                    if node:
                        layer_list_flat.append(node)
                        l.append(node)
                layer_list.append(l)
        layer_array = np.array(layer_list_flat)
        layer_array = layer_array.reshape([-1, 1])
        logger.info("TDM Tree max layer: {}".format(len(layer_list)))
        logger.info("TDM Tree layer_node_num_list: {}".format(
            [len(i) for i in layer_list]))
        return layer_array, layer_list

    def tdm_info_prepare(self):
        """load tdm tree param from list file"""
        info_array = np.load(self.tree_info_path)
        return info_array
