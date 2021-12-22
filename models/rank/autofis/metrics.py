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

import paddle.nn.functional
from paddle.metric import Metric


class LogLoss(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def compute(self, pred, label, *args):
        return paddle.nn.functional.log_loss(pred, label).mean()

    def update(self, pred, label, *args):
        self.loss += self.compute(pred, label).item()
        self.n += 1
        return

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.loss = 0
        self.n = 0

    def accumulate(self):
        """
        Computes and returns the accumulated metric.
        """
        return self.loss / self.n

    def _init_name(self, name):
        name = name or 'log_loss'
        self._name = [name]

    def name(self):
        """
        Return name of metric instance.
        """
        return self._name
