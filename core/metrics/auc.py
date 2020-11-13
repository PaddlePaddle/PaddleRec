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
import paddle
import paddle.fluid as fluid

from paddlerec.core.metric import Metric
from paddle.fluid.layers.tensor import Variable


class AUC(Metric):
    """
    Metric For Fluid Model
    """

    def __init__(self,
                 input,
                 label,
                 curve='ROC',
                 num_thresholds=2**12 - 1,
                 topk=1,
                 slide_steps=1):
        """ """
        if not isinstance(input, Variable):
            raise ValueError("input must be Variable, but received %s" %
                             type(input))
        if not isinstance(label, Variable):
            raise ValueError("label must be Variable, but received %s" %
                             type(label))

        auc_out, batch_auc_out, [
            batch_stat_pos, batch_stat_neg, stat_pos, stat_neg
        ] = fluid.layers.auc(input,
                             label,
                             curve=curve,
                             num_thresholds=num_thresholds,
                             topk=topk,
                             slide_steps=slide_steps)

        prob = paddle.slice(input, axes=[1], starts=[1], ends=[2])
        label_cast = paddle.cast(label, dtype="float32")
        label_cast.stop_gradient = True
        sqrerr, abserr, prob, q, pos, total = \
            fluid.contrib.layers.ctr_metric_bundle(prob, label_cast)

        self._global_metric_state_vars = dict()
        self._global_metric_state_vars['stat_pos'] = (stat_pos.name, "float32")
        self._global_metric_state_vars['stat_neg'] = (stat_neg.name, "float32")
        self._global_metric_state_vars['total_ins_num'] = (total.name,
                                                           "float32")
        self._global_metric_state_vars['pos_ins_num'] = (pos.name, "float32")
        self._global_metric_state_vars['q'] = (q.name, "float32")
        self._global_metric_state_vars['prob'] = (prob.name, "float32")
        self._global_metric_state_vars['abserr'] = (abserr.name, "float32")
        self._global_metric_state_vars['sqrerr'] = (sqrerr.name, "float32")

        self.metrics = dict()
        self.metrics["AUC"] = auc_out
        self.metrics["BATCH_AUC"] = batch_auc_out

    def _calculate_bucket_error(self, global_pos, global_neg):
        """R
        """
        num_bucket = len(global_pos)
        last_ctr = -1.0
        impression_sum = 0.0
        ctr_sum = 0.0
        click_sum = 0.0
        error_sum = 0.0
        error_count = 0.0
        click = 0.0
        show = 0.0
        ctr = 0.0
        adjust_ctr = 0.0
        relative_error = 0.0
        actual_ctr = 0.0
        relative_ctr_error = 0.0
        k_max_span = 0.01
        k_relative_error_bound = 0.05
        for i in range(num_bucket):
            click = global_pos[i]
            show = global_pos[i] + global_neg[i]
            ctr = float(i) / num_bucket
            if abs(ctr - last_ctr) > k_max_span:
                last_ctr = ctr
                impression_sum = 0.0
                ctr_sum = 0.0
                click_sum = 0.0
            impression_sum += show
            ctr_sum += ctr * show
            click_sum += click
            if impression_sum == 0:
                continue
            adjust_ctr = ctr_sum / impression_sum
            if adjust_ctr == 0:
                continue
            relative_error = \
                math.sqrt((1 - adjust_ctr) / (adjust_ctr * impression_sum))
            if relative_error < k_relative_error_bound:
                actual_ctr = click_sum / impression_sum
                relative_ctr_error = abs(actual_ctr / adjust_ctr - 1)
                error_sum += relative_ctr_error * impression_sum
                error_count += impression_sum
                last_ctr = -1

        bucket_error = error_sum / error_count if error_count > 0 else 0.0
        return bucket_error

    def _calculate_auc(self, global_pos, global_neg):
        """R
        """
        num_bucket = len(global_pos)
        area = 0.0
        pos = 0.0
        neg = 0.0
        new_pos = 0.0
        new_neg = 0.0
        total_ins_num = 0
        for i in range(num_bucket):
            index = num_bucket - 1 - i
            new_pos = pos + global_pos[index]
            total_ins_num += global_pos[index]
            new_neg = neg + global_neg[index]
            total_ins_num += global_neg[index]
            area += (new_neg - neg) * (pos + new_pos) / 2
            pos = new_pos
            neg = new_neg
        auc_value = None
        if pos * neg == 0 or total_ins_num == 0:
            auc_value = 0.5
        else:
            auc_value = area / (pos * neg)
        return auc_value

    def _calculate(self, global_metrics):
        result = dict()
        for key in self._global_metric_state_vars:
            if key not in global_metrics:
                raise ValueError("%s not existed" % key)
            result[key] = global_metrics[key][0]

        if result['total_ins_num'] == 0:
            result['auc'] = 0
            result['bucket_error'] = 0
            result['actual_ctr'] = 0
            result['predict_ctr'] = 0
            result['mae'] = 0
            result['rmse'] = 0
            result['copc'] = 0
            result['mean_q'] = 0
        else:
            result['auc'] = self._calculate_auc(result['stat_pos'],
                                                result['stat_neg'])
            result['bucket_error'] = self._calculate_bucket_error(
                result['stat_pos'], result['stat_neg'])
            result['actual_ctr'] = result['pos_ins_num'] / result[
                'total_ins_num']
            result['mae'] = result['abserr'] / result['total_ins_num']
            result['rmse'] = math.sqrt(result['sqrerr'] /
                                       result['total_ins_num'])
            result['predict_ctr'] = result['prob'] / result['total_ins_num']
            if abs(result['predict_ctr']) > 1e-6:
                result['copc'] = result['actual_ctr'] / result['predict_ctr']
            result['mean_q'] = result['q'] / result['total_ins_num']

        result_str = "AUC=%.6f BUCKET_ERROR=%.6f MAE=%.6f RMSE=%.6f " \
                     "Actural_CTR=%.6f Predicted_CTR=%.6f COPC=%.6f MEAN Q_VALUE=%.6f Ins number=%s" % \
                     (result['auc'], result['bucket_error'], result['mae'], result['rmse'],
                      result['actual_ctr'],
                      result['predict_ctr'], result['copc'], result['mean_q'], result['total_ins_num'])
        return result_str

    def get_result(self):
        return self.metrics
