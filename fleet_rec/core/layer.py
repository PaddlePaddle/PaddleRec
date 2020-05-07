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


class Layer(object):
    """R
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        """R
        """
        pass

    def generate(self, mode, param):
        """R
        """
        if mode == 'fluid':
            return self.generate_fluid(param)
        elif mode == 'tensorflow':
            return self.generate_tensorflow(param)
        print('unsupport this mode: ' + mode)
        return None, None

    @abc.abstractmethod
    def generate_fluid(self, param):
        """R
        """
        pass

    def generate_tensorflow(self, param):
        """ Not implement currently
        """
        pass
