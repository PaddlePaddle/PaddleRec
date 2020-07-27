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


class PrecisionRecall(Metric):
    """
    Metric For Fluid Model
    """

    def __init__(self, **kwargs):
        """ """
        helper = LayerHelper("PaddleRec_PrecisionRecall", **kwargs)
        predict = kwargs.get("input")
        origin_label = kwargs.get("label")
        label = fluid.layers.cast(origin_label, dtype="int32")
        label.stop_gradient = True
        num_cls = kwargs.get("class_num")
        max_probs, indices = fluid.layers.nn.topk(predict, k=1)
        indices = fluid.layers.cast(indices, dtype="int32")
        indices.stop_gradient = True

        states_info, _ = helper.create_or_get_global_variable(
            name="states_info",
            persistable=True,
            dtype='float32',
            shape=[num_cls, 4])
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
            shape=[num_cls, 4], value=0.0, dtype="float32")
        batch_states.stop_gradient = True

        helper.append_op(
            type="precision_recall",
            attrs={'class_number': num_cls},
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

        self._need_clear_list = [("states_info", "float32")]

        self.metrics = dict()
        self.metrics["precision_recall_f1"] = accum_metrics
        self.metrics["accum_states"] = states_info

    # self.metrics["batch_metrics"] = batch_metrics

    def get_result(self):
        return self.metrics
