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
from paddlerec.core.metrics import PrecisionRecall
import paddle
import paddle.fluid as fluid


def calc_precision(tp_count, fp_count):
    if tp_count > 0.0 or fp_count > 0.0:
        return tp_count / (tp_count + fp_count)
    return 1.0


def calc_recall(tp_count, fn_count):
    if tp_count > 0.0 or fn_count > 0.0:
        return tp_count / (tp_count + fn_count)
    return 1.0


def calc_f1_score(precision, recall):
    if precision > 0.0 or recall > 0.0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def get_states(idxs, labels, cls_num, weights=None, batch_nums=1):
    ins_num = idxs.shape[0]
    # TP FP TN FN
    states = np.zeros((cls_num, 4)).astype('float32')
    for i in range(ins_num):
        w = weights[i] if weights is not None else 1.0
        idx = idxs[i][0]
        label = labels[i][0]
        if idx == label:
            states[idx][0] += w
            for j in range(cls_num):
                states[j][2] += w
            states[idx][2] -= w
        else:
            states[label][3] += w
            states[idx][1] += w
            for j in range(cls_num):
                states[j][2] += w
            states[label][2] -= w
            states[idx][2] -= w
    return states


def compute_metrics(states, cls_num):
    total_tp_count = 0.0
    total_fp_count = 0.0
    total_fn_count = 0.0
    macro_avg_precision = 0.0
    macro_avg_recall = 0.0
    for i in range(cls_num):
        total_tp_count += states[i][0]
        total_fp_count += states[i][1]
        total_fn_count += states[i][3]
        macro_avg_precision += calc_precision(states[i][0], states[i][1])
        macro_avg_recall += calc_recall(states[i][0], states[i][3])
    metrics = []
    macro_avg_precision /= cls_num
    macro_avg_recall /= cls_num
    metrics.append(macro_avg_precision)
    metrics.append(macro_avg_recall)
    metrics.append(calc_f1_score(macro_avg_precision, macro_avg_recall))
    micro_avg_precision = calc_precision(total_tp_count, total_fp_count)
    metrics.append(micro_avg_precision)
    micro_avg_recall = calc_recall(total_tp_count, total_fn_count)
    metrics.append(micro_avg_recall)
    metrics.append(calc_f1_score(micro_avg_precision, micro_avg_recall))
    return np.array(metrics).astype('float32')


class TestPrecisionRecall(unittest.TestCase):
    def setUp(self):
        self.ins_num = 64
        self.cls_num = 10
        self.batch_nums = 3

        self.datas = []
        self.states = np.zeros((self.cls_num, 4)).astype('float32')

        for i in range(self.batch_nums):
            probs = np.random.uniform(0, 1.0, (self.ins_num,
                                               self.cls_num)).astype('float32')
            idxs = np.array(np.argmax(
                probs, axis=1)).reshape(self.ins_num, 1).astype('int32')
            labels = np.random.choice(range(self.cls_num),
                                      self.ins_num).reshape(
                                          (self.ins_num, 1)).astype('int32')
            self.datas.append((probs, labels))
            states = get_states(idxs, labels, self.cls_num)
            self.states = np.add(self.states, states)
        self.metrics = compute_metrics(self.states, self.cls_num)

        self.place = fluid.core.CPUPlace()

    def build_network(self):
        predict = fluid.data(
            name="predict",
            shape=[-1, self.cls_num],
            dtype='float32',
            lod_level=0)
        label = fluid.data(
            name="label", shape=[-1, 1], dtype='int32', lod_level=0)

        precision_recall = PrecisionRecall(
            input=predict, label=label, class_num=self.cls_num)
        return precision_recall

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
            outs = exe.run(
                fluid.default_main_program(),
                feed={'predict': self.datas[i][0],
                      'label': self.datas[i][1]},
                fetch_list=fetch_vars,
                return_numpy=True)

        outs = dict(zip(metric_keys, outs))
        self.assertTrue(np.allclose(outs['accum_states'], self.states))
        self.assertTrue(np.allclose(outs['precision_recall_f1'], self.metrics))


if __name__ == '__main__':
    unittest.main()
