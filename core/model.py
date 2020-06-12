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

import abc
import os
import paddle.fluid as fluid

from paddlerec.core.utils import envs


class ModelBase(object):
    """Base Model
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        """R
        """
        self._cost = None
        self._metrics = {}
        self._data_var = []
        self._infer_data_var = []
        self._infer_results = {}
        self._data_loader = None
        self._infer_data_loader = None
        self._fetch_interval = 20
        self._platform = envs.get_platform()
        self._init_hyper_parameters()
        self._env = config
        self._slot_inited = False

    def _init_hyper_parameters(self):
        pass

    def _init_slots(self, **kargs):
        if self._slot_inited:
            return
        self._slot_inited = True
        dataset = {}
        model_dict = {}
        for i in self._env["phase"]:
            if i["name"] == kargs["name"]:
                model_dict = i
                break
        for i in self._env["dataset"]:
            if i["name"] == model_dict["dataset_name"]:
                dataset = i
                break
        name = "dataset." + dataset["name"] + "."
        sparse_slots = envs.get_global_env(name + "sparse_slots", "").strip()
        dense_slots = envs.get_global_env(name + "dense_slots", "").strip()
        if sparse_slots != "" or dense_slots != "":
            if sparse_slots == "":
                sparse_slots = []
            else:
                sparse_slots = sparse_slots.strip().split(" ")
            if dense_slots == "":
                dense_slots = []
            else:
                dense_slots = dense_slots.strip().split(" ")
            dense_slots_shape = [[
                int(j) for j in i.split(":")[1].strip("[]").split(",")
            ] for i in dense_slots]
            dense_slots = [i.split(":")[0] for i in dense_slots]
            self._dense_data_var = []
            for i in range(len(dense_slots)):
                l = fluid.layers.data(
                    name=dense_slots[i],
                    shape=dense_slots_shape[i],
                    dtype="float32")
                self._data_var.append(l)
                self._dense_data_var.append(l)
            self._sparse_data_var = []
            for name in sparse_slots:
                l = fluid.layers.data(
                    name=name, shape=[1], lod_level=1, dtype="int64")
                self._data_var.append(l)
                self._sparse_data_var.append(l)

        dataset_class = envs.get_global_env(name + "type")
        if dataset_class == "DataLoader":
            self._init_dataloader()

    def _init_dataloader(self, is_infer=False):
        if is_infer:
            data = self._infer_data_var
        else:
            data = self._data_var
        self._data_loader = fluid.io.DataLoader.from_generator(
            feed_list=data,
            capacity=64,
            use_double_buffer=False,
            iterable=False)

    def get_inputs(self):
        return self._data_var

    def get_infer_inputs(self):
        return self._infer_data_var

    def get_infer_results(self):
        return self._infer_results

    def get_avg_cost(self):
        """R
        """
        return self._cost

    def get_metrics(self):
        """R
        """
        return self._metrics

    def get_fetch_period(self):
        return self._fetch_interval

    def _build_optimizer(self, name, lr, strategy=None):
        name = name.upper()
        optimizers = ["SGD", "ADAM", "ADAGRAD"]
        if name not in optimizers:
            raise ValueError(
                "configured optimizer can only supported SGD/Adam/Adagrad")

        if name == "SGD":
            os.environ["FLAGS_communicator_is_sgd_optimizer"] = '1'
        else:
            os.environ["FLAGS_communicator_is_sgd_optimizer"] = '0'

        if name == "SGD":
            optimizer_i = fluid.optimizer.SGD(lr)
        elif name == "ADAM":
            optimizer_i = fluid.optimizer.Adam(lr, lazy_mode=True)
        elif name == "ADAGRAD":
            optimizer_i = fluid.optimizer.Adagrad(lr)
        else:
            raise ValueError(
                "configured optimizer can only supported SGD/Adam/Adagrad")

        return optimizer_i

    def optimizer(self):
        opt_name = envs.get_global_env("hyper_parameters.optimizer.class")
        opt_lr = envs.get_global_env(
            "hyper_parameters.optimizer.learning_rate")
        opt_strategy = envs.get_global_env(
            "hyper_parameters.optimizer.strategy")

        return self._build_optimizer(opt_name, opt_lr, opt_strategy)

    def input_data(self, is_infer=False, **kwargs):
        name = "dataset." + kwargs.get("dataset_name") + "."
        sparse_slots = envs.get_global_env(name + "sparse_slots", "").strip()
        dense_slots = envs.get_global_env(name + "dense_slots", "").strip()
        self._sparse_data_var_map = {}
        self._dense_data_var_map = {}
        if sparse_slots != "" or dense_slots != "":
            if sparse_slots == "":
                sparse_slots = []
            else:
                sparse_slots = sparse_slots.strip().split(" ")
            if dense_slots == "":
                dense_slots = []
            else:
                dense_slots = dense_slots.strip().split(" ")
            dense_slots_shape = [[
                int(j) for j in i.split(":")[1].strip("[]").split(",")
            ] for i in dense_slots]
            dense_slots = [i.split(":")[0] for i in dense_slots]
            self._dense_data_var = []
            data_var_ = []
            for i in range(len(dense_slots)):
                l = fluid.layers.data(
                    name=dense_slots[i],
                    shape=dense_slots_shape[i],
                    dtype="float32")
                data_var_.append(l)
                self._dense_data_var.append(l)
                self._dense_data_var_map[dense_slots[i]] = l
            self._sparse_data_var = []
            for name in sparse_slots:
                l = fluid.layers.data(
                    name=name, shape=[1], lod_level=1, dtype="int64")
                data_var_.append(l)
                self._sparse_data_var.append(l)
                self._sparse_data_var_map[name] = l
            return data_var_

        else:
            return None

    def net(self, is_infer=False):
        return None

    def train_net(self):
        pass

    def infer_net(self):
        pass
