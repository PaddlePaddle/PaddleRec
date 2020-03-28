
import math
import numpy as np
import paddle.fluid as fluid
from .base import Metric


class AUCMetric(Metric):
    """
    Metric For Paddle Model
    """

    def __init__(self, config, fleet):
        """ """
        self.config = config
        self.fleet = fleet

    def clear(self, scope, params):
        """
        Clear current metric value, usually set to zero
        Args:
            scope : paddle runtime var container
            params(dict) :
                label : a group name for metric
                metric_dict : current metric_items in group
        Return:
            None
        """
        self._label = params['label']
        self._metric_dict = params['metric_dict']
        self._result = {}
        place = fluid.CPUPlace()
        for metric_name in self._metric_dict:
            metric_config = self._metric_dict[metric_name]
            if scope.find_var(metric_config['var'].name) is None:
                continue
            metric_var = scope.var(metric_config['var'].name).get_tensor()
            data_type = 'float32'
            if 'data_type' in metric_config:
                data_type = metric_config['data_type']
            data_array = np.zeros(metric_var._get_dims()).astype(data_type)
            metric_var.set(data_array, place)

    def get_metric(self, scope, metric_name):
        """
        reduce metric named metric_name from all worker
        Return:
            metric reduce result
        """
        metric = np.array(scope.find_var(metric_name).get_tensor())
        old_metric_shape = np.array(metric.shape)
        metric = metric.reshape(-1)
        global_metric = np.copy(metric) * 0
        self.fleet._role_maker._node_type_comm.Allreduce(metric, global_metric)
        global_metric = global_metric.reshape(old_metric_shape)
        return global_metric[0]

    def get_global_metrics(self, scope, metric_dict):
        """
        reduce all metric in metric_dict from all worker
        Return:
            dict : {matric_name : metric_result}
        """
        self.fleet._role_maker._barrier_worker()
        result = {}
        for metric_name in metric_dict:
            metric_item = metric_dict[metric_name]
            if scope.find_var(metric_item['var'].name) is None:
                result[metric_name] = None
                continue
            result[metric_name] = self.get_metric(scope, metric_item['var'].name)
        return result

    def calculate_auc(self, global_pos, global_neg):
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

    def calculate_bucket_error(self, global_pos, global_neg):
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

    def calculate(self, scope, params):
        """ """
        self._label = params['label']
        self._metric_dict = params['metric_dict']
        self.fleet._role_maker._barrier_worker()
        result = self.get_global_metrics(scope, self._metric_dict)
        if result['total_ins_num'] == 0:
            self._result = result
            self._result['auc'] = 0
            self._result['bucket_error'] = 0
            self._result['actual_ctr'] = 0
            self._result['predict_ctr'] = 0
            self._result['mae'] = 0
            self._result['rmse'] = 0
            self._result['copc'] = 0
            self._result['mean_q'] = 0
            return self._result
        if 'stat_pos' in result and 'stat_neg' in result:
            result['auc'] = self.calculate_auc(result['stat_pos'], result['stat_neg'])
            result['bucket_error'] = self.calculate_auc(result['stat_pos'], result['stat_neg'])
        if 'pos_ins_num' in result:
            result['actual_ctr'] = result['pos_ins_num'] / result['total_ins_num']
        if 'abserr' in result:
            result['mae'] = result['abserr'] / result['total_ins_num']
        if 'sqrerr' in result:
            result['rmse'] = math.sqrt(result['sqrerr'] / result['total_ins_num'])
        if 'prob' in result:
            result['predict_ctr'] = result['prob'] / result['total_ins_num']
            if abs(result['predict_ctr']) > 1e-6:
                result['copc'] = result['actual_ctr'] / result['predict_ctr']

        if 'q' in result:
            result['mean_q'] = result['q'] / result['total_ins_num']
        self._result = result
        return result

    def get_result(self):
        """ """
        return self._result

    def get_result_to_string(self):
        """ """
        result = self.get_result()
        result_str = "%s AUC=%.6f BUCKET_ERROR=%.6f MAE=%.6f RMSE=%.6f " \
                     "Actural_CTR=%.6f Predicted_CTR=%.6f COPC=%.6f MEAN Q_VALUE=%.6f Ins number=%s" % \
                     (self._label, result['auc'], result['bucket_error'], result['mae'], result['rmse'],
                      result['actual_ctr'],
                      result['predict_ctr'], result['copc'], result['mean_q'], result['total_ins_num'])
        return result_str
