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
from paddlerec.core.metrics import RecallK
import paddle
import paddle.fluid as fluid


class TestRecallK(unittest.TestCase):
    def setUp(self):
        self.ins_num = 64
        self.cls_num = 10
        self.topk = 2
        self.batch_nums = 3

        self.datas = []
        self.match_num = 0.0
        for i in range(self.batch_nums):
            z = np.random.uniform(0, 1.0, (self.ins_num,
                                           self.cls_num)).astype('float32')
            pred = np.exp(z) / sum(np.exp(z))
            label = np.random.choice(range(self.cls_num),
                                     self.ins_num).reshape(
                                         (self.ins_num, 1)).astype('int64')
            self.datas.append((pred, label))
            max_k_preds = pred.argsort(
                axis=1)[:, -self.topk:][:, ::-1]  #top-k label
            match_array = np.logical_or.reduce(max_k_preds == label, axis=1)
            self.match_num += np.sum(match_array).astype('float32')
        self.place = fluid.core.CPUPlace()

    def build_network(self):
        pred = fluid.data(
            name="pred",
            shape=[-1, self.cls_num],
            dtype='float32',
            lod_level=0)

        label = fluid.data(
            name="label", shape=[-1, 1], dtype='int64', lod_level=0)

        recall_k = RecallK(input=pred, label=label, k=self.topk)
        return recall_k

    def test_forward(self):
        net = self.build_network()
        metrics = net.get_result()
        fetch_vars = []
        metric_keys = []
        for item in metrics.items():
            fetch_vars.append(item[1])
            metric_keys.append(item[0])

        exe = fluid.Executor(self.place)
        exe.run(fluid.default_startup_program())
        for i in range(self.batch_nums):
            outs = exe.run(
                fluid.default_main_program(),
                feed={'pred': self.datas[i][0],
                      'label': self.datas[i][1]},
                fetch_list=fetch_vars,
                return_numpy=True)

        outs = dict(zip(metric_keys, outs))
        self.assertTrue(
            np.allclose(outs['ins_cnt'], self.ins_num * self.batch_nums))
        self.assertTrue(np.allclose(outs['pos_cnt'], self.match_num))
        self.assertTrue(
            np.allclose(outs['Recall@%d_ACC' % (self.topk)],
                        np.array(self.match_num / (self.ins_num *
                                                   self.batch_nums))))


if __name__ == '__main__':
    unittest.main()
