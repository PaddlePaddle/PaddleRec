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

from __future__ import print_function
import os
import sys
import warnings
import logging
import numpy as np
import math
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
import common

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_global_auc(scope=fluid.global_scope(),
                   stat_pos="_generated_var_2",
                   stat_neg="_generated_var_3"):
    """
        Get global auc of all distributed workers.

        Args:
            scope(Scope): Scope object, default is fluid.global_scope()
            stat_pos(str): name of auc pos bucket Variable
            stat_neg(str): name of auc neg bucket Variable

        Returns:
            auc_value(float), total_ins_num(int)

        """
    if scope.find_var(stat_pos) is None or scope.find_var(stat_neg) is None:
        logger.info("not found auc bucket")
        return None
    fleet.barrier_worker()
    # auc pos bucket
    pos = np.array(scope.find_var(stat_pos).get_tensor())
    # auc pos bucket shape
    old_pos_shape = np.array(pos.shape)
    # reshape to one dim
    pos = pos.reshape(-1)
    #global_pos = np.copy(pos) * 0
    # mpi allreduce
    global_pos = fleet.util.all_reduce(pos)
    # reshape to its original shape
    global_pos = global_pos.reshape(old_pos_shape)
    # print('debug global auc global_pos: ', global_pos)

    # auc neg bucket
    neg = np.array(scope.find_var(stat_neg).get_tensor())
    old_neg_shape = np.array(neg.shape)
    neg = neg.reshape(-1)
    #global_neg = np.copy(neg) * 0
    global_neg = fleet.util.all_reduce(neg)
    global_neg = global_neg.reshape(old_neg_shape)
    # print('debug global auc global_neg: ', global_neg)

    # calculate auc
    num_bucket = len(global_pos[0])
    area = 0.0
    pos = 0.0
    neg = 0.0
    new_pos = 0.0
    new_neg = 0.0
    total_ins_num = 0
    for i in range(num_bucket):
        index = num_bucket - 1 - i
        new_pos = pos + global_pos[0][index]
        total_ins_num += global_pos[0][index]
        new_neg = neg + global_neg[0][index]
        total_ins_num += global_neg[0][index]
        area += (new_neg - neg) * (pos + new_pos) / 2
        pos = new_pos
        neg = new_neg

    if pos * neg == 0 or total_ins_num == 0:
        auc_value = 0.5
    else:
        auc_value = area / (pos * neg)

    fleet.barrier_worker()
    return auc_value


def get_global_metrics(scope=fluid.global_scope(),
                       stat_pos_name="_generated_var_2",
                       stat_neg_name="_generated_var_3",
                       sqrerr_name="sqrerr",
                       abserr_name="abserr",
                       prob_name="prob",
                       q_name="q",
                       pos_ins_num_name="pos",
                       total_ins_num_name="total"):
    from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
    fleet_util = FleetUtil()
    if scope.find_var(stat_pos_name) is None or \
            scope.find_var(stat_neg_name) is None:
        fleet_util.rank0_print("not found auc bucket")
        return [None] * 9
    elif scope.find_var(sqrerr_name) is None:
        fleet_util.rank0_print("not found sqrerr_name=%s" % sqrerr_name)
        return [None] * 9
    elif scope.find_var(abserr_name) is None:
        fleet_util.rank0_print("not found abserr_name=%s" % abserr_name)
        return [None] * 9
    elif scope.find_var(prob_name) is None:
        fleet_util.rank0_print("not found prob_name=%s" % prob_name)
        return [None] * 9
    elif scope.find_var(q_name) is None:
        fleet_util.rank0_print("not found q_name=%s" % q_name)
        return [None] * 9
    elif scope.find_var(pos_ins_num_name) is None:
        fleet_util.rank0_print("not found pos_ins_num_name=%s" %
                               pos_ins_num_name)
        return [None] * 9
    elif scope.find_var(total_ins_num_name) is None:
        fleet_util.rank0_print("not found total_ins_num_name=%s" % \
                               total_ins_num_name)
        return [None] * 9

    # barrier worker to ensure all workers finished training
    fleet.barrier_worker()

    # get auc
    auc = get_global_auc(scope, stat_pos_name, stat_neg_name)
    pos = np.array(scope.find_var(stat_pos_name).get_tensor())
    # auc pos bucket shape
    old_pos_shape = np.array(pos.shape)
    # reshape to one dim
    pos = pos.reshape(-1)
    global_pos = np.copy(pos) * 0
    # mpi allreduce
    # fleet._role_maker._all_reduce(pos, global_pos)
    global_pos = fleet.util.all_reduce(pos)
    # reshape to its original shape
    global_pos = global_pos.reshape(old_pos_shape)
    # auc neg bucket
    neg = np.array(scope.find_var(stat_neg_name).get_tensor())
    old_neg_shape = np.array(neg.shape)
    neg = neg.reshape(-1)
    global_neg = np.copy(neg) * 0
    # fleet._role_maker._all_reduce(neg, global_neg)
    global_neg = fleet.util.all_reduce(neg)
    global_neg = global_neg.reshape(old_neg_shape)

    num_bucket = len(global_pos[0])

    def get_metric(name):
        metric = np.array(scope.find_var(name).get_tensor())
        old_metric_shape = np.array(metric.shape)
        metric = metric.reshape(-1)
        # print(name, 'ori value:', metric)
        global_metric = np.copy(metric) * 0
        # fleet._role_maker._all_reduce(metric, global_metric)
        global_metric = fleet.util.all_reduce(metric)
        global_metric = global_metric.reshape(old_metric_shape)
        # print(name, global_metric)
        return global_metric[0]

    global_sqrerr = get_metric(sqrerr_name)
    global_abserr = get_metric(abserr_name)
    global_prob = get_metric(prob_name)
    global_q_value = get_metric(q_name)
    # note: get ins_num from auc bucket is not actual value,
    # so get it from metric op
    pos_ins_num = get_metric(pos_ins_num_name)
    total_ins_num = get_metric(total_ins_num_name)
    neg_ins_num = total_ins_num - pos_ins_num

    mae = global_abserr / total_ins_num
    rmse = math.sqrt(global_sqrerr / total_ins_num)
    return_actual_ctr = pos_ins_num / total_ins_num
    predicted_ctr = global_prob / total_ins_num
    mean_predict_qvalue = global_q_value / total_ins_num
    copc = 0.0
    if abs(predicted_ctr > 1e-6):
        copc = return_actual_ctr / predicted_ctr

    # calculate bucket error
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
        click = global_pos[0][i]
        show = global_pos[0][i] + global_neg[0][i]
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

    return [
        auc, bucket_error, mae, rmse, return_actual_ctr, predicted_ctr, copc,
        mean_predict_qvalue, int(total_ins_num)
    ]


def get_global_metrics_str(scope, metric_list, prefix):
    if len(metric_list) != 10:
        raise ValueError("len(metric_list) != 10, %s" % len(metric_list))

    auc, bucket_error, mae, rmse, actual_ctr, predicted_ctr, copc, \
    mean_predict_qvalue, total_ins_num = get_global_metrics( \
        scope, metric_list[2].name, metric_list[3].name, metric_list[4].name, metric_list[5].name, \
        metric_list[6].name, metric_list[7].name, metric_list[8].name, metric_list[9].name)
    metrics_str = "%s global AUC=%.6f BUCKET_ERROR=%.6f MAE=%.6f " \
                    "RMSE=%.6f Actural_CTR=%.6f Predicted_CTR=%.6f " \
                    "COPC=%.6f MEAN Q_VALUE=%.6f Ins number=%s" % \
                    (prefix, auc, bucket_error, mae, rmse, \
                    actual_ctr, predicted_ctr, copc, mean_predict_qvalue, \
                    total_ins_num)
    return metrics_str


def clear_metrics(scope, var_list, var_types):
    from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
    fleet_util = FleetUtil()
    for i in range(len(var_list)):
        fleet_util.set_zero(var_list[i].name, scope, param_type=var_types[i])
