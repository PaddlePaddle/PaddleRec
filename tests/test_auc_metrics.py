#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from paddlerec.core.metrics import AUC
import paddle
import paddle.fluid as fluid


class TestAUC(unittest.TestCase):
    def setUp(self):
        self.ins_num = 64
        self.batch_nums = 3
        self.probs = np.random.uniform(0, 1.0,
                                       (self.ins_num, 2)).astype('float32')
        self.labels = np.random.choice(range(2), self.ins_num).reshape(
            (self.ins_num, 1)).astype('int64')

        self.place = fluid.core.CPUPlace()

        self.num_thresholds = 2**12
        python_auc = fluid.metrics.Auc(name="auc",
                                       curve='ROC',
                                       num_thresholds=self.num_thresholds)
        for i in range(self.batch_nums):
            python_auc.update(self.probs, self.labels)

        self.auc = np.array(python_auc.eval())

    def build_network(self):
        predict = fluid.data(
            name="predict", shape=[-1, 2], dtype='float32', lod_level=0)
        label = fluid.data(
            name="label", shape=[-1, 1], dtype='int64', lod_level=0)

        auc = AUC(input=predict,
                  label=label,
                  num_thresholds=self.num_thresholds,
                  curve='ROC')
        return auc

    def test_forward(self):
        precision_recall = self.build_network()
        metrics = precision_recall.get_result()
        fetch_vars = []
        metric_keys = []
        for item in metrics.items():
            fetch_vars.append(item[1])
            metric_keys.append(item[0])

        exe = fluid.Executor(self.place)
        exe.run(fluid.default_startup_program())
        for i in range(self.batch_nums):
            outs = exe.run(fluid.default_main_program(),
                           feed={'predict': self.probs,
                                 'label': self.labels},
                           fetch_list=fetch_vars,
                           return_numpy=True)

        outs = dict(zip(metric_keys, outs))
        self.assertTrue(np.allclose(outs['AUC'], self.auc))


if __name__ == '__main__':
    unittest.main()
