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


class Metric(object):
    """R
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        """ """
        pass

    @abc.abstractmethod
    def clear(self, scope, params):
        """
        clear current value
        Args:
            scope: value container
            params: extend varilable for clear
        """
        pass

    @abc.abstractmethod
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

    @abc.abstractmethod
    def __str__(self):
        """
        Return:
            result(string) : calculate result with string format, for output
        """
        pass
