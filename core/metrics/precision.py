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


class Precision(Metric):
    """
    Metric For Fluid Model
    """

    def __init__(self, **kwargs):
        """ """
        helper = LayerHelper("PaddleRec_Precision", **kwargs)
        self.batch_accuracy = accuracy(
            kwargs.get("input"), kwargs.get("label"), kwargs.get("k"))
        local_ins_num, _ = helper.create_or_get_global_variable(
            name="local_ins_num", persistable=True, dtype='float32',
            shape=[1])
        local_pos_num, _ = helper.create_or_get_global_variable(
            name="local_pos_num", persistable=True, dtype='float32',
            shape=[1])

        batch_pos_num, _ = helper.create_or_get_global_variable(
            name="batch_pos_num",
            persistable=False,
            dtype='float32',
            shape=[1])
        batch_ins_num, _ = helper.create_or_get_global_variable(
            name="batch_ins_num",
            persistable=False,
            dtype='float32',
            shape=[1])

        tmp_ones = helper.create_global_variable(
            name="batch_size_like_ones",
            persistable=False,
            dtype='float32',
            shape=[-1])

        for var in [
                batch_pos_num, batch_ins_num, local_pos_num, local_ins_num
        ]:
            print(var, type(var))
            helper.set_variable_initializer(
                var, Constant(
                    value=0.0, force_cpu=True))

        helper.append_op(
            type='fill_constant_batch_size_like',
            inputs={"Input": kwargs.get("label")},
            outputs={'Out': [tmp_ones]},
            attrs={
                'shape': [-1, 1],
                'dtype': tmp_ones.dtype,
                'value': float(1.0),
            })
        helper.append_op(
            type="reduce_sum",
            inputs={"X": [tmp_ones]},
            outputs={"Out": [batch_ins_num]})

        helper.append_op(
            type="elementwise_mul",
            inputs={"X": [batch_ins_num],
                    "Y": [self.batch_accuracy]},
            outputs={"Out": [batch_pos_num]})

        helper.append_op(
            type="elementwise_add",
            inputs={"X": [local_pos_num],
                    "Y": [batch_pos_num]},
            outputs={"Out": [local_pos_num]})

        helper.append_op(
            type="elementwise_add",
            inputs={"X": [local_ins_num],
                    "Y": [batch_ins_num]},
            outputs={"Out": [local_ins_num]})

        self.accuracy = local_pos_num / local_ins_num

        self.metrics = dict()
        metric_varname = "P@%d" % kwargs.get("k")
        self.metrics[metric_varname] = self.accuracy

    def get_result(self):
        return self.metrics
