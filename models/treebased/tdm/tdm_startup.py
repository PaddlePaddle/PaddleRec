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
Training use fluid with DistributeTranspiler.
"""
from __future__ import print_function

import time
import logging

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddlerec.core.utils import envs
from paddlerec.core.trainers.framework.startup import StartupBase
from paddlerec.core.trainer import EngineMode

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)
special_param = ["TDM_Tree_Travel", "TDM_Tree_Layer", "TDM_Tree_Info"]


class Startup(StartupBase):
    def startup(self, context):
        logger.info("Run TDM Trainer Startup Pass")
        if context["engine"] == EngineMode.SINGLE:
            self._single_startup(context)
        else:
            self._cluster_startup(context)

        context['status'] = 'train_pass'

    def _single_startup(self, context):
        load_tree_from_numpy = envs.get_global_env(
            "hyper_parameters.tree.load_tree_from_numpy", False)
        model_dict = context("env")["phase"][0]
        with fluid.scope_guard(context["model"][model_dict["name"]]["scope"]):
            context["exe"].run(context["model"][model_dict["name"]][
                "startup_program"])
            if load_tree_from_numpy:
                logger.info("load tree from numpy")

                self.tree_layer_path = envs.get_global_env(
                    "hyper_parameters.tree.tree_layer_path", "")

                self.tree_travel_path = envs.get_global_env(
                    "hyper_parameters.tree.tree_travel_path", "")

                self.tree_info_path = envs.get_global_env(
                    "hyper_parameters.tree.tree_info_path", "")

                self.tree_emb_path = envs.get_global_env(
                    "hyper_parameters.tree.tree_emb_path",
                    "", )

                for param_name in special_param:
                    param_t = fluid.global_scope().find_var(
                        param_name).get_tensor()
                    param_array = self._tdm_prepare(param_name)
                    if param_name == 'TDM_Tree_Emb':
                        param_t.set(
                            param_array.astype('float32'), context["place"])
                    else:
                        param_t.set(
                            param_array.astype('int32'), context["place"])

                logger.info("Begin Save Init model.")
                fluid.io.save_persistables(
                    executor=context["exe"],
                    main_program=context["model"][model_dict["name"]][
                        "main_program"],
                    dirname="./init_model")
                logger.info("End Save Init model.")

            load_paddle_model = envs.get_global_env(
                "hyper_parameters.tree.load_paddle_model", False)
            assert load_tree_from_numpy != load_paddle_model, "Please Don't use load_tree_from_numpy & load_paddle_model at the same time"
            warmup_model_path = envs.get_global_env(
                "runner." + context["runner_name"] + ".init_model_path", None)
            if load_paddle_model:
                # 从paddle二进制模型加载参数
                assert warmup_model_path != None, "set runner.init_model_path for loading model"
                fluid.io.load_persistables(
                    executor=context["exe"],
                    dirname=warmup_model_path,
                    main_program=context["model"][model_dict["name"]][
                        "main_program"])
                logger.info("Load persistables from \"{}\"".format(
                    warmup_model_path))

    def _cluster_startup(self, context):
        warmup_model_path = envs.get_global_env(
            "runner." + context["runner_name"] + ".init_model_path", None)
        assert warmup_model_path != None, "set runner.init_model_path for loading model"
        model_dict = context("env")["phase"][0]
        with fluid.scope_guard(context["model"][model_dict["name"]]["scope"]):
            context["exe"].run(context["model"][model_dict["name"]][
                "startup_program"])

            def is_tdm_tree_var(var):
                res = var.name in special_param
                return res

            fluid.io.load_vars(
                context["exe"],
                dirname=warmup_model_path,
                main_program=context["model"][model_dict["name"]][
                    "main_program"],
                predicate=is_tdm_tree_var)

    """ --------  tree file load detail  --------- """

    def _tdm_prepare(self, param_name):
        if param_name == "TDM_Tree_Travel":
            travel_array = self._tdm_travel_prepare()
            return travel_array
        elif param_name == "TDM_Tree_Layer":
            layer_array, _ = self._tdm_layer_prepare()
            return layer_array
        elif param_name == "TDM_Tree_Info":
            info_array = self._tdm_info_prepare()
            return info_array
        else:
            raise " {} is not a special tdm param name".format(param_name)

    def _tdm_travel_prepare(self):
        """load tdm tree param from npy/list file"""
        travel_array = np.load(self.tree_travel_path)
        logger.info("TDM Tree leaf node nums: {}".format(travel_array.shape[
            0]))
        return travel_array

    def _tdm_layer_prepare(self):
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

    def _tdm_info_prepare(self):
        """load tdm tree param from list file"""
        info_array = np.load(self.tree_info_path)
        return info_array
