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

import math

import numpy as np
import paddle.fluid as fluid

from paddlerec.core.metric import Metric
from paddle.fluid.layers import accuracy
from paddle.fluid.initializer import Constant
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers.tensor import Variable


class RecallK(Metric):
    """
    Metric For Fluid Model
    """

    def __init__(self, **kwargs):
        """ """
        if "input" not in kwargs or "label" not in kwargs:
            raise ValueError("RecallK expect input and label as inputs.")
        predict = kwargs.get('input')
        label = kwargs.get('label')
        self.k = kwargs.get("k", 20)

        if not isinstance(predict, Variable):
            raise ValueError("input must be Variable, but received %s" %
                             type(predict))
        if not isinstance(label, Variable):
            raise ValueError("label must be Variable, but received %s" %
                             type(label))

        helper = LayerHelper("PaddleRec_RecallK", **kwargs)
        batch_accuracy = accuracy(predict, label, self.k)
        global_ins_cnt, _ = helper.create_or_get_global_variable(
            name="ins_cnt", persistable=True, dtype='float32', shape=[1])
        global_pos_cnt, _ = helper.create_or_get_global_variable(
            name="pos_cnt", persistable=True, dtype='float32', shape=[1])

        for var in [global_ins_cnt, global_pos_cnt]:
            helper.set_variable_initializer(
                var, Constant(
                    value=0.0, force_cpu=True))

        tmp_ones = fluid.layers.fill_constant(
            shape=fluid.layers.shape(label), dtype="float32", value=1.0)
        batch_ins = fluid.layers.reduce_sum(tmp_ones)
        batch_pos = batch_ins * batch_accuracy

        helper.append_op(
            type="elementwise_add",
            inputs={"X": [global_ins_cnt],
                    "Y": [batch_ins]},
            outputs={"Out": [global_ins_cnt]})

        helper.append_op(
            type="elementwise_add",
            inputs={"X": [global_pos_cnt],
                    "Y": [batch_pos]},
            outputs={"Out": [global_pos_cnt]})

        self.acc = global_pos_cnt / global_ins_cnt

        self._global_communicate_var = dict()
        self._global_communicate_var['ins_cnt'] = (global_ins_cnt.name,
                                                   "float32")
        self._global_communicate_var['pos_cnt'] = (global_pos_cnt.name,
                                                   "float32")

        metric_name = "Acc(Recall@%d)" % self.k
        self.metrics = dict()
        self.metrics["InsCnt"] = global_ins_cnt
        self.metrics["RecallCnt"] = global_pos_cnt
        self.metrics[metric_name] = self.acc

    # self.metrics["batch_metrics"] = batch_metrics
    def calculate(self, global_metrics):
        for key in self._global_communicate_var:
            if key not in global_metrics:
                raise ValueError("%s not existed" % key)
        ins_cnt = global_metrics['ins_cnt'][0]
        pos_cnt = global_metrics['pos_cnt'][0]
        if ins_cnt == 0:
            acc = 0
        else:
            acc = float(pos_cnt) / ins_cnt
        return "InsCnt=%s RecallCnt=%s Acc(Recall@%d)=%s" % (
            str(ins_cnt), str(pos_cnt), self.k, str(acc))

    def get_result(self):
        return self.metrics
