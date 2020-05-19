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

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet

from paddlerec.core.utils import envs
from paddlerec.core.trainers.cluster_trainer import ClusterTrainer


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)
special_param = ["TDM_Tree_Travel", "TDM_Tree_Layer", "TDM_Tree_Info"]


class TDMClusterTrainer(ClusterTrainer):
    def server(self, context):
        namespace = "train.startup"
        init_model_path = envs.get_global_env(
            "cluster.init_model_path", "", namespace)
        assert init_model_path != "", "Cluster train must has init_model for TDM"
        fleet.init_server(init_model_path)
        logger.info("TDM: load model from {}".format(init_model_path))
        fleet.run_server()
        context['is_exit'] = True

    def startup(self, context):
        self._exe.run(fleet.startup_program)

        namespace = "train.startup"
        load_tree = envs.get_global_env(
            "tree.load_tree", True, namespace)
        self.tree_layer_path = envs.get_global_env(
            "tree.tree_layer_path", "", namespace)
        self.tree_travel_path = envs.get_global_env(
            "tree.tree_travel_path", "", namespace)
        self.tree_info_path = envs.get_global_env(
            "tree.tree_info_path", "", namespace)

        save_init_model = envs.get_global_env(
            "cluster.save_init_model", False, namespace)
        init_model_path = envs.get_global_env(
            "cluster.init_model_path", "", namespace)

        if load_tree:
            # covert tree to tensor, set it into Fluid's variable.
            for param_name in special_param:
                param_t = fluid.global_scope().find_var(param_name).get_tensor()
                param_array = self._tdm_prepare(param_name)
                param_t.set(param_array.astype('int32'), self._place)

        if save_init_model:
            logger.info("Begin Save Init model.")
            fluid.io.save_persistables(
                executor=self._exe, dirname=init_model_path)
            logger.info("End Save Init model.")

        context['status'] = 'train_pass'

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
        logger.info("TDM Tree leaf node nums: {}".format(
            travel_array.shape[0]))
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
