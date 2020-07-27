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


class AUC(Metric):
    """
    Metric For Fluid Model
    """

    def __init__(self, **kwargs):
        """ """
        predict = kwargs.get("input")
        label = kwargs.get("label")
        curve = kwargs.get("curve", 'ROC')
        num_thresholds = kwargs.get("num_thresholds", 2**12 - 1)
        topk = kwargs.get("topk", 1)
        slide_steps = kwargs.get("slide_steps", 1)
        auc_out, batch_auc_out, [
            batch_stat_pos, batch_stat_neg, stat_pos, stat_neg
        ] = fluid.layers.auc(predict,
                             label,
                             curve=curve,
                             num_thresholds=num_thresholds,
                             topk=topk,
                             slide_steps=slide_steps)

        self._need_clear_list = [(stat_pos.name, "float32"),
                                 (stat_neg.name, "float32")]
        self.metrics = dict()
        self.metrics["AUC"] = auc_out
        self.metrics["BATCH_AUC"] = batch_auc_out

    def get_result(self):
        return self.metrics
