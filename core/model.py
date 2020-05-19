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
    """R
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

    def _build_optimizer(self, name, lr):
        name = name.upper()
        optimizers = ["SGD", "ADAM", "ADAGRAD"]
        if name not in optimizers:
            raise ValueError(
                "configured optimizer can only supported SGD/Adam/Adagrad")

        if name == "SGD":
            reg = envs.get_global_env(
                "hyper_parameters.reg", 0.0001, self._namespace)
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
        learning_rate = envs.get_global_env(
            "hyper_parameters.learning_rate", None, self._namespace)
        optimizer = envs.get_global_env(
            "hyper_parameters.optimizer", None, self._namespace)
        print(">>>>>>>>>>>.learnig rate: %s" % learning_rate)
        return self._build_optimizer(optimizer, learning_rate)

    @abc.abstractmethod
    def train_net(self):
        """R
        """
        pass

    @abc.abstractmethod
    def infer_net(self):
        pass
