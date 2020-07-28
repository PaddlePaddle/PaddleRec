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
from paddlerec.core.metrics import PosNegRatio
import paddle
import paddle.fluid as fluid


class TestPosNegRatio(unittest.TestCase):
    def setUp(self):
        self.ins_num = 64
        self.batch_nums = 3

        self.datas = []
        self.right_cnt = 0.0
        self.wrong_cnt = 0.0
        for i in range(self.batch_nums):
            neg_score = np.random.uniform(0, 1.0,
                                          (self.ins_num, 1)).astype('float32')
            pos_score = np.random.uniform(0, 1.0,
                                          (self.ins_num, 1)).astype('float32')

            right_cnt = np.sum(np.less(neg_score, pos_score)).astype('int32')
            wrong_cnt = np.sum(np.less_equal(pos_score, neg_score)).astype(
                'int32')
            self.right_cnt += float(right_cnt)
            self.wrong_cnt += float(wrong_cnt)
            self.datas.append((pos_score, neg_score))

        self.place = fluid.core.CPUPlace()

    def build_network(self):
        pos_score = fluid.data(
            name="pos_score", shape=[-1, 1], dtype='float32', lod_level=0)

        neg_score = fluid.data(
            name="neg_score", shape=[-1, 1], dtype='float32', lod_level=0)

        pairwise_pn = PosNegRatio(pos_score=pos_score, neg_score=neg_score)
        return pairwise_pn

    def test_forward(self):
        pn = self.build_network()
        metrics = pn.get_result()
        fetch_vars = []
        metric_keys = []
        for item in metrics.items():
            fetch_vars.append(item[1])
            metric_keys.append(item[0])

        exe = fluid.Executor(self.place)
        exe.run(fluid.default_startup_program())
        for i in range(self.batch_nums):
            outs = exe.run(fluid.default_main_program(),
                           feed={
                               'pos_score': self.datas[i][0],
                               'neg_score': self.datas[i][1]
                           },
                           fetch_list=fetch_vars,
                           return_numpy=True)

        outs = dict(zip(metric_keys, outs))
        self.assertTrue(np.allclose(outs['RightCnt'], self.right_cnt))
        self.assertTrue(np.allclose(outs['WrongCnt'], self.wrong_cnt))
        self.assertTrue(
            np.allclose(outs['PN'],
                        np.array((self.right_cnt + 1.0) / (self.wrong_cnt + 1.0
                                                           ))))

    def test_exception(self):
        self.assertRaises(Exception, PosNegRatio)
        self.assertRaises(
            Exception,
            PosNegRatio,
            pos_score=self.datas[0][0],
            neg_score=self.datas[0][1]),


if __name__ == '__main__':
    unittest.main()
