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
from paddle.fluid.layers import nn, accuracy
from paddle.fluid.initializer import Constant
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers.tensor import Variable


class PrecisionRecall(Metric):
    """
    Metric For Fluid Model
    """

    def __init__(self, **kwargs):
        """ """
        if "input" not in kwargs or "label" not in kwargs or "class_num" not in kwargs:
            raise ValueError(
                "PrecisionRecall expect input, label and class_num as inputs.")
        predict = kwargs.get("input")
        label = kwargs.get("label")
        self.num_cls = kwargs.get("class_num")

        if not isinstance(predict, Variable):
            raise ValueError("input must be Variable, but received %s" %
                             type(predict))
        if not isinstance(label, Variable):
            raise ValueError("label must be Variable, but received %s" %
                             type(label))

        helper = LayerHelper("PaddleRec_PrecisionRecall", **kwargs)
        label = fluid.layers.cast(label, dtype="int32")
        label.stop_gradient = True
        max_probs, indices = fluid.layers.nn.topk(predict, k=1)
        indices = fluid.layers.cast(indices, dtype="int32")
        indices.stop_gradient = True

        states_info, _ = helper.create_or_get_global_variable(
            name="states_info",
            persistable=True,
            dtype='float32',
            shape=[self.num_cls, 4])
        states_info.stop_gradient = True

        helper.set_variable_initializer(
            states_info, Constant(
                value=0.0, force_cpu=True))

        batch_metrics, _ = helper.create_or_get_global_variable(
            name="batch_metrics",
            persistable=False,
            dtype='float32',
            shape=[6])
        accum_metrics, _ = helper.create_or_get_global_variable(
            name="global_metrics",
            persistable=False,
            dtype='float32',
            shape=[6])

        batch_states = fluid.layers.fill_constant(
            shape=[self.num_cls, 4], value=0.0, dtype="float32")
        batch_states.stop_gradient = True

        helper.append_op(
            type="precision_recall",
            attrs={'class_number': self.num_cls},
            inputs={
                'MaxProbs': [max_probs],
                'Indices': [indices],
                'Labels': [label],
                'StatesInfo': [states_info]
            },
            outputs={
                'BatchMetrics': [batch_metrics],
                'AccumMetrics': [accum_metrics],
                'AccumStatesInfo': [batch_states]
            })
        helper.append_op(
            type="assign",
            inputs={'X': [batch_states]},
            outputs={'Out': [states_info]})

        batch_states.stop_gradient = True
        states_info.stop_gradient = True

        self._global_communicate_var = dict()
        self._global_communicate_var['states_info'] = (states_info.name,
                                                       "float32")

        self.metrics = dict()
        self.metrics["precision_recall_f1"] = accum_metrics
        self.metrics["[TP FP TN FN]"] = states_info

    # self.metrics["batch_metrics"] = batch_metrics

    def calculate(self, global_metrics):
        for key in self._global_communicate_var:
            if key not in global_metrics:
                raise ValueError("%s not existed" % key)

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

        states = global_metrics["states_info"]
        total_tp_count = 0.0
        total_fp_count = 0.0
        total_fn_count = 0.0
        macro_avg_precision = 0.0
        macro_avg_recall = 0.0
        for i in range(self.num_cls):
            total_tp_count += states[i][0]
            total_fp_count += states[i][1]
            total_fn_count += states[i][3]
            macro_avg_precision += calc_precision(states[i][0], states[i][1])
            macro_avg_recall += calc_recall(states[i][0], states[i][3])
        metrics = []
        macro_avg_precision /= self.num_cls
        macro_avg_recall /= self.num_cls
        metrics.append(macro_avg_precision)
        metrics.append(macro_avg_recall)
        metrics.append(calc_f1_score(macro_avg_precision, macro_avg_recall))
        micro_avg_precision = calc_precision(total_tp_count, total_fp_count)
        metrics.append(micro_avg_precision)
        micro_avg_recall = calc_recall(total_tp_count, total_fn_count)
        metrics.append(micro_avg_recall)
        metrics.append(calc_f1_score(micro_avg_precision, micro_avg_recall))
        return "total metrics: [TP, FP, TN, FN]=%s; precision_recall_f1=%s" % (
            str(states), str(np.array(metrics).astype('float32')))

    def get_result(self):
        return self.metrics
