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

import paddle.fluid as fluid

from paddlerec.core.utils import envs


class Model(object):
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
        self._namespace = "train.model"
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
        for i in self._env["executor"]:
            if i["name"] == kargs["name"]:
                model_dict = i
                break
        for i in self._env["dataset"]:
            if i["name"] == model_dict["dataset_name"]:
                dataset = i
                break
        name = "dataset." + dataset["name"] + "."
        sparse_slots = envs.get_global_env(name + "sparse_slots")
        dense_slots = envs.get_global_env(name + "dense_slots")
        if sparse_slots is not None or dense_slots is not None:
            sparse_slots = sparse_slots.strip().split(" ")
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

        dataset_class = dataset["type"]
        if dataset_class == "DataLoader":
            self._init_dataloader()

    def _init_dataloader(self):
        self._data_loader = fluid.io.DataLoader.from_generator(
            feed_list=self._data_var,
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
            reg = envs.get_global_env("hyper_parameters.reg", 0.0001,
                                      self._namespace)
            optimizer_i = fluid.optimizer.SGD(
                lr, regularization=fluid.regularizer.L2DecayRegularizer(reg))
        elif name == "ADAM":
            optimizer_i = fluid.optimizer.Adam(lr, lazy_mode=True)
        elif name == "ADAGRAD":
            optimizer_i = fluid.optimizer.Adagrad(lr)
        else:
            raise ValueError(
                "configured optimizer can only supported SGD/Adam/Adagrad")

        return optimizer_i

    def optimizer(self):
        learning_rate = envs.get_global_env("hyper_parameters.learning_rate",
                                            None, self._namespace)
        optimizer = envs.get_global_env("hyper_parameters.optimizer", None,
                                        self._namespace)
        return self._build_optimizer(optimizer, learning_rate)

    def input_data(self, is_infer=False, **kwargs):
        name = "dataset." + kwargs.get("dataset_name") + "."
        sparse_slots = envs.get_global_env(name + "sparse_slots")
        dense_slots = envs.get_global_env(name + "dense_slots")
        if sparse_slots is not None or dense_slots is not None:
            sparse_slots = sparse_slots.strip().split(" ")
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
            self._sparse_data_var = []
            for name in sparse_slots:
                l = fluid.layers.data(
                    name=name, shape=[1], lod_level=1, dtype="int64")
                data_var_.append(l)
                self._sparse_data_var.append(l)
            return data_var_

        else:
            return None

    def net(self, is_infer=False):
        return None

    def _construct_reader(self, is_infer=False):
        if is_infer:
            self._infer_data_loader = fluid.io.DataLoader.from_generator(
                feed_list=self._infer_data_var,
                capacity=64,
                use_double_buffer=False,
                iterable=False)
        else:
            dataset_class = envs.get_global_env("dataset_class", None,
                                                "train.reader")
            if dataset_class == "DataLoader":
                self._data_loader = fluid.io.DataLoader.from_generator(
                    feed_list=self._data_var,
                    capacity=64,
                    use_double_buffer=False,
                    iterable=False)

    def train_net(self):
        input_data = self.input_data(is_infer=False)
        self._data_var = input_data
        self._construct_reader(is_infer=False)
        self.net(input_data, is_infer=False)

    def infer_net(self):
        input_data = self.input_data(is_infer=True)
        self._infer_data_var = input_data
        self._construct_reader(is_infer=True)
        self.net(input_data, is_infer=True)
