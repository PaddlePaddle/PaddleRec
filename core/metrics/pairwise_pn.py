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


class PosNegRatio(Metric):
    """
    Metric For Fluid Model
    """

    def __init__(self, **kwargs):
        """ """
        helper = LayerHelper("PaddleRec_PosNegRatio", **kwargs)
        if "pos_score" not in kwargs or "neg_score" not in kwargs:
            raise ValueError(
                "PosNegRatio expect pos_score and neg_score as inputs.")
        pos_score = kwargs.get('pos_score')
        neg_score = kwargs.get('neg_score')

        if not isinstance(pos_score, Variable):
            raise ValueError("pos_score must be Variable, but received %s" %
                             type(pos_score))
        if not isinstance(neg_score, Variable):
            raise ValueError("neg_score must be Variable, but received %s" %
                             type(neg_score))

        wrong = fluid.layers.cast(
            fluid.layers.less_equal(pos_score, neg_score), dtype='float32')
        wrong_cnt = fluid.layers.reduce_sum(wrong)
        right = fluid.layers.cast(
            fluid.layers.less_than(neg_score, pos_score), dtype='float32')
        right_cnt = fluid.layers.reduce_sum(right)

        global_right_cnt, _ = helper.create_or_get_global_variable(
            name="right_cnt", persistable=True, dtype='float32', shape=[1])
        global_wrong_cnt, _ = helper.create_or_get_global_variable(
            name="wrong_cnt", persistable=True, dtype='float32', shape=[1])

        for var in [global_right_cnt, global_wrong_cnt]:
            helper.set_variable_initializer(
                var, Constant(
                    value=0.0, force_cpu=True))

        helper.append_op(
            type="elementwise_add",
            inputs={"X": [global_right_cnt],
                    "Y": [right_cnt]},
            outputs={"Out": [global_right_cnt]})
        helper.append_op(
            type="elementwise_add",
            inputs={"X": [global_wrong_cnt],
                    "Y": [wrong_cnt]},
            outputs={"Out": [global_wrong_cnt]})
        self.pn = (global_right_cnt + 1.0) / (global_wrong_cnt + 1.0)

        self._need_clear_list = [("right_cnt", "float32"),
                                 ("wrong_cnt", "float32")]

        self.metrics = dict()
        self.metrics['wrong_cnt'] = global_wrong_cnt
        self.metrics['right_cnt'] = global_right_cnt
        self.metrics['pos_neg_ratio'] = self.pn

    def get_result(self):
        return self.metrics
