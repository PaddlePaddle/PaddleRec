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
import numpy as np


class Metric(object):
    """R
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        """R
        """
        pass

    def clear(self, scope=None):
        """R
        """
        if scope is None:
            scope = fluid.global_scope()

        place = fluid.CPUPlace()
        for key in self._global_metric_state_vars:
            varname, dtype = self._global_metric_state_vars[key]
            var = scope.find_var(varname)
            if not var:
                continue
            var = var.get_tensor()
            data_array = np.zeros(var._get_dims()).astype(dtype)
            var.set(data_array, place)

    def _get_global_metric_state(self, fleet, scope, metric_name, mode="sum"):
        """R
        """
        var = scope.find_var(metric_name)
        if not var:
            return None
        input = np.array(var.get_tensor())
        if fleet is None:
            return input
        fleet._role_maker._barrier_worker()
        old_shape = np.array(input.shape)
        input = input.reshape(-1)
        output = np.copy(input) * 0
        fleet._role_maker._all_reduce(input, output, mode=mode)
        output = output.reshape(old_shape)
        return output

    def calc_global_metrics(self, fleet, scope=None):
        """R
        """
        if scope is None:
            scope = fluid.global_scope()

        global_metrics = dict()
        for key in self._global_metric_state_vars:
            varname, dtype = self._global_metric_state_vars[key]
            global_metrics[key] = self._get_global_metric_state(fleet, scope,
                                                                varname)

        return self._calculate(global_metrics)

    def _calculate(self, global_metrics):
        pass

    @abc.abstractmethod
    def get_result(self):
        """
        Return:
            result(dict) : calculate result
        """
        pass

    def __str__(self):
        """
        Return:
            result(string) : calculate result with string format, for output
        """
        pass
