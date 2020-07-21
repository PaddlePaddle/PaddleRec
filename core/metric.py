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
        """ """
        pass

    def clear(self, scope=None, **kwargs):
        """
        clear current value
        Args:
            scope: value container
            params: extend varilable for clear
        """
        if scope is None:
            scope = fluid.global_scope()

        place = fluid.CPUPlace()
        for (varname, dtype) in self._need_clear_list:
            if scope.find_var(varname) is None:
                continue
            var = scope.var(varname).get_tensor()
            data_array = np.zeros(var._get_dims()).astype(dtype)
            var.set(data_array, place)

    def calculate(self, scope, params):
        """
        calculate result
        Args:
            scope: value container
            params: extend varilable for clear
        """
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
