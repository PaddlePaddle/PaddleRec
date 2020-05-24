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
Training use fluid with DistributeTranspiler
"""
import os
import time
import logging
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddlerec.core.utils import envs
from paddlerec.core.trainers.general_trainer import GeneralTrainer, EngineMode

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)
special_param = ["TDM_Tree_Travel", "TDM_Tree_Layer", "TDM_Tree_Info"]


class TDMTrainer(GeneralTrainer):
    def startup(self, context):
        logger.info("Run TDM Trainer Startup Pass")
        if self.engine == EngineMode.SINGLE:
            self._single_startup(context)
        else:
            self._cluster_startup(context)

        context['status'] = 'train_pass'

    def _single_startup(self, context):
        namespace = "train.startup"
        load_tree_from_numpy = envs.get_global_env(
            "single.load_tree_from_numpy", False, namespace)
        self._exe.run(self._startup_program)
        if load_tree_from_numpy:
            logger.info("load tree from numpy")

            self.tree_layer_path = envs.get_global_env("tree.tree_layer_path",
                                                       "", namespace)

            self.tree_travel_path = envs.get_global_env(
                "tree.tree_travel_path", "", namespace)

            self.tree_info_path = envs.get_global_env("tree.tree_info_path",
                                                      "", namespace)

            self.tree_emb_path = envs.get_global_env("tree.tree_emb_path", "",
                                                     namespace)
            for param_name in special_param:
                param_t = fluid.global_scope().find_var(param_name).get_tensor(
                )
                param_array = self._tdm_prepare(param_name)
                if param_name == 'TDM_Tree_Emb':
                    param_t.set(param_array.astype('float32'), self._place)
                else:
                    param_t.set(param_array.astype('int32'), self._place)

        load_paddle_model = envs.get_global_env("single.load_paddle_model",
                                                False, namespace)
        assert load_tree_from_numpy != load_paddle_model, "Please Don't use load_tree_from_numpy & load_paddle_model at the same time"

        if load_paddle_model:
            # 从paddle二进制模型加载参数
            warmup_model_path = envs.get_global_env("warmup_model_path", None,
                                                    "train.trainer")
            assert warmup_model_path != None, "set train.trainer.warmup_model_path for loading model"
            fluid.io.load_persistables(
                executor=self._exe,
                dirname=warmup_model_path,
                main_program=self._main_program)
            logger.info("Load persistables from \"{}\"".format(
                warmup_model_path))

        save_init_model = envs.get_global_env("single.save_init_model", False,
                                              namespace)
        init_model_path = envs.get_global_env("single.init_model_path", "",
                                              namespace)
        if save_init_model:
            logger.info("Begin Save Init model.")
            fluid.io.save_persistables(
                executor=self._exe, dirname=init_model_path)
            logger.info("End Save Init model.")

    def _cluster_startup(self, context):
        warmup_model_path = envs.get_global_env("warmup_model_path", None,
                                                "train.trainer")
        assert warmup_model_path != None, "set train.trainer.warmup_model_path for loading model"
        self._exe.run(self._startup_program)

        def is_tdm_tree_var(var):
            res = var.name in special_param
            return res

        fluid.io.load_vars(
            self._exe,
            dirname=warmup_model_path,
            main_program=self._main_program,
            predicate=is_tdm_tree_var)

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
