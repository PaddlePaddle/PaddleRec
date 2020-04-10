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


import sys
import time
import json
import datetime
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
from paddle.fluid.incubate.fleet.base.role_maker import GeneralRoleMaker

from fleetrec.utils import fs as fs
from fleetrec.utils import util as util
from fleetrec.metrics.auc_metrics import AUCMetric
from fleetrec.models import base as model_basic
from fleetrec.reader import dataset
from .trainer import Trainer


def wroker_numric_opt(value, env, opt):
    """
    numric count opt for workers
    Args:
        value: value for count
        env: mpi/gloo
        opt: count operator, SUM/MAX/MIN/AVG
    Return:
        count result
    """
    local_value = np.array([value])
    global_value = np.copy(local_value) * 0
    fleet._role_maker.all_reduce_worker(local_value, global_value, opt)
    return global_value[0]


def worker_numric_sum(value, env="mpi"):
    """R
    """
    return wroker_numric_opt(value, env, "sum")


def worker_numric_avg(value, env="mpi"):
    """R
    """
    return worker_numric_sum(value, env) / fleet.worker_num()


def worker_numric_min(value, env="mpi"):
    """R
    """
    return wroker_numric_opt(value, env, "min")


def worker_numric_max(value, env="mpi"):
    """R
    """
    return wroker_numric_opt(value, env, "max")


class CtrPaddleTrainer(Trainer):
    """R
    """

    def __init__(self, config):
        """R
        """
        Trainer.__init__(self, config)
        config['output_path'] = util.get_absolute_path(
            config['output_path'], config['io']['afs'])

        self.global_config = config
        self._metrics = {}

        self._path_generator = util.PathGenerator({
            'templates': [
                {'name': 'xbox_base_done', 'template': config['output_path'] + '/xbox_base_done.txt'},
                {'name': 'xbox_delta_done', 'template': config['output_path'] + '/xbox_patch_done.txt'},
                {'name': 'xbox_base', 'template': config['output_path'] + '/xbox/{day}/base/'},
                {'name': 'xbox_delta', 'template': config['output_path'] + '/xbox/{day}/delta-{pass_id}/'},
                {'name': 'batch_model', 'template': config['output_path'] + '/batch_model/{day}/{pass_id}/'}
            ]
        })
        if 'path_generator' in config:
            self._path_generator.add_path_template(config['path_generator'])

        self.regist_context_processor('uninit', self.init)
        self.regist_context_processor('startup', self.startup)
        self.regist_context_processor('begin_day', self.begin_day)
        self.regist_context_processor('train_pass', self.train_pass)
        self.regist_context_processor('end_day', self.end_day)

    def init(self, context):
        """R
        """
        role_maker = None
        if self.global_config.get('process_mode', 'mpi') == 'brilliant_cpu':
            afs_config = self.global_config['io']['afs']
            role_maker = GeneralRoleMaker(
                hdfs_name=afs_config['fs_name'], hdfs_ugi=afs_config['fs_ugi'],
                path=self.global_config['output_path'] + "/gloo",
                init_timeout_seconds=1200, run_timeout_seconds=1200)
        fleet.init(role_maker)
        data_var_list = []
        data_var_name_dict = {}
        runnnable_scope = []
        runnnable_cost_op = []
        context['status'] = 'startup'

        for executor in self.global_config['executor']:
            scope = fluid.Scope()
            self._exector_context[executor['name']] = {}
            self._exector_context[executor['name']]['scope'] = scope
            self._exector_context[executor['name']]['model'] = model_basic.create(executor)
            model = self._exector_context[executor['name']]['model']
            self._metrics.update(model.get_metrics())
            runnnable_scope.append(scope)
            runnnable_cost_op.append(model.get_cost_op())
            for var in model._data_var:
                if var.name in data_var_name_dict:
                    continue
                data_var_list.append(var)
                data_var_name_dict[var.name] = var

        optimizer = model_basic.YamlModel.build_optimizer({
            'metrics': self._metrics,
            'optimizer_conf': self.global_config['optimizer']
        })
        optimizer.minimize(runnnable_cost_op, runnnable_scope)
        for executor in self.global_config['executor']:
            scope = self._exector_context[executor['name']]['scope']
            model = self._exector_context[executor['name']]['model']
            program = model._build_param['model']['train_program']
            if not executor['is_update_sparse']:
                program._fleet_opt["program_configs"][str(id(model.get_cost_op().block.program))]["push_sparse"] = []
            if 'train_thread_num' not in executor:
                executor['train_thread_num'] = self.global_config['train_thread_num']
            with fluid.scope_guard(scope):
                self._exe.run(model._build_param['model']['startup_program'])
            model.dump_model_program('./')

        # server init done
        if fleet.is_server():
            return 0

        self._dataset = {}
        for dataset_item in self.global_config['dataset']['data_list']:
            dataset_item['data_vars'] = data_var_list
            dataset_item.update(self.global_config['io']['afs'])
            dataset_item["batch_size"] = self.global_config['batch_size']
            self._dataset[dataset_item['name']] = dataset.FluidTimeSplitDataset(dataset_item)
        # if config.need_reqi_changeslot and config.reqi_dnn_plugin_day >= last_day and config.reqi_dnn_plugin_pass >= last_pass:
        #    util.reqi_changeslot(config.hdfs_dnn_plugin_path, join_save_params, common_save_params, update_save_params, scope2, scope3)
        fleet.init_worker()
        pass

    def print_log(self, log_str, params):
        """R
        """
        params['index'] = fleet.worker_index()
        if params['master']:
            if fleet.worker_index() == 0:
                print(log_str)
                sys.stdout.flush()
        else:
            print(log_str)
        if 'stdout' in params:
            params['stdout'] += str(datetime.datetime.now()) + log_str

    def print_global_metrics(self, scope, model, monitor_data, stdout_str):
        """R
        """
        metrics = model.get_metrics()
        metric_calculator = AUCMetric(None)
        for metric in metrics:
            metric_param = {'label': metric, 'metric_dict': metrics[metric]}
            metric_calculator.calculate(scope, metric_param)
            metric_result = metric_calculator.get_result_to_string()
            self.print_log(metric_result, {'master': True, 'stdout': stdout_str})
            monitor_data += metric_result
            metric_calculator.clear(scope, metric_param)

    def save_model(self, day, pass_index, base_key):
        """R
        """
        cost_printer = util.CostPrinter(util.print_cost,
                                        {'master': True, 'log_format': 'save model cost %s sec'})
        model_path = self._path_generator.generate_path('batch_model', {'day': day, 'pass_id': pass_index})
        save_mode = 0  # just save all
        if pass_index < 1:  # batch_model
            save_mode = 3  # unseen_day++, save all
        util.rank0_print("going to save_model %s" % model_path)
        fleet.save_persistables(None, model_path, mode=save_mode)
        if fleet._role_maker.is_first_worker():
            self._train_pass.save_train_progress(day, pass_index, base_key, model_path, is_checkpoint=True)
        cost_printer.done()
        return model_path

    def save_xbox_model(self, day, pass_index, xbox_base_key, monitor_data):
        """R
        """
        stdout_str = ""
        xbox_patch_id = str(int(time.time()))
        util.rank0_print("begin save delta model")

        model_path = ""
        xbox_model_donefile = ""
        cost_printer = util.CostPrinter(util.print_cost, {'master': True, \
                                                          'log_format': 'save xbox model cost %s sec',
                                                          'stdout': stdout_str})
        if pass_index < 1:
            save_mode = 2
            xbox_patch_id = xbox_base_key
            model_path = self._path_generator.generate_path('xbox_base', {'day': day})
            xbox_model_donefile = self._path_generator.generate_path('xbox_base_done', {'day': day})
        else:
            save_mode = 1
            model_path = self._path_generator.generate_path('xbox_delta', {'day': day, 'pass_id': pass_index})
            xbox_model_donefile = self._path_generator.generate_path('xbox_delta_done', {'day': day})
        total_save_num = fleet.save_persistables(None, model_path, mode=save_mode)
        cost_printer.done()

        cost_printer = util.CostPrinter(util.print_cost, {'master': True,
                                                          'log_format': 'save cache model cost %s sec',
                                                          'stdout': stdout_str})
        model_file_handler = fs.FileHandler(self.global_config['io']['afs'])
        if self.global_config['save_cache_model']:
            cache_save_num = fleet.save_cache_model(None, model_path, mode=save_mode)
            model_file_handler.write(
                "file_prefix:part\npart_num:16\nkey_num:%d\n" % cache_save_num,
                model_path + '/000_cache/sparse_cache.meta', 'w')
        cost_printer.done()
        util.rank0_print("save xbox cache model done, key_num=%s" % cache_save_num)

        save_env_param = {
            'executor': self._exe,
            'save_combine': True
        }
        cost_printer = util.CostPrinter(util.print_cost, {'master': True,
                                                          'log_format': 'save dense model cost %s sec',
                                                          'stdout': stdout_str})
        if fleet._role_maker.is_first_worker():
            for executor in self.global_config['executor']:
                if 'layer_for_inference' not in executor:
                    continue
                executor_name = executor['name']
                model = self._exector_context[executor_name]['model']
                save_env_param['inference_list'] = executor['layer_for_inference']
                save_env_param['scope'] = self._exector_context[executor_name]['scope']
                model.dump_inference_param(save_env_param)
                for dnn_layer in executor['layer_for_inference']:
                    model_file_handler.cp(dnn_layer['save_file_name'],
                                          model_path + '/dnn_plugin/' + dnn_layer['save_file_name'])
        fleet._role_maker._barrier_worker()
        cost_printer.done()

        xbox_done_info = {
            "id": xbox_patch_id,
            "key": xbox_base_key,
            "ins_path": "",
            "ins_tag": "feasign",
            "partition_type": "2",
            "record_count": "111111",
            "monitor_data": monitor_data,
            "mpi_size": str(fleet.worker_num()),
            "input": model_path.rstrip("/") + "/000",
            "job_id": util.get_env_value("JOB_ID"),
            "job_name": util.get_env_value("JOB_NAME")
        }
        if fleet._role_maker.is_first_worker():
            model_file_handler.write(json.dumps(xbox_done_info) + "\n", xbox_model_donefile, 'a')
            if pass_index > 0:
                self._train_pass.save_train_progress(day, pass_index, xbox_base_key, model_path, is_checkpoint=False)
        fleet._role_maker._barrier_worker()
        return stdout_str

    def run_executor(self, executor_config, dataset, stdout_str):
        """R
        """
        day = self._train_pass.date()
        pass_id = self._train_pass._pass_id
        xbox_base_key = self._train_pass._base_key
        executor_name = executor_config['name']
        scope = self._exector_context[executor_name]['scope']
        model = self._exector_context[executor_name]['model']
        with fluid.scope_guard(scope):
            util.rank0_print("Begin " + executor_name + " pass")
            begin = time.time()
            program = model._build_param['model']['train_program']
            self._exe.train_from_dataset(program, dataset, scope,
                                         thread=executor_config['train_thread_num'], debug=self.global_config['debug'])
            end = time.time()
            local_cost = (end - begin) / 60.0
            avg_cost = worker_numric_avg(local_cost)
            min_cost = worker_numric_min(local_cost)
            max_cost = worker_numric_max(local_cost)
            util.rank0_print("avg train time %s mins, min %s mins, max %s mins" % (avg_cost, min_cost, max_cost))
            self._exector_context[executor_name]['cost'] = max_cost

            monitor_data = ""
            self.print_global_metrics(scope, model, monitor_data, stdout_str)
            util.rank0_print("End " + executor_name + " pass")
            if self._train_pass.need_dump_inference(pass_id) and executor_config['dump_inference_model']:
                stdout_str += self.save_xbox_model(day, pass_id, xbox_base_key, monitor_data)
        fleet._role_maker._barrier_worker()

    def startup(self, context):
        """R
        """
        if fleet.is_server():
            fleet.run_server()
            context['status'] = 'wait'
            return
        stdout_str = ""
        self._train_pass = util.TimeTrainPass(self.global_config)
        if not self.global_config['cold_start']:
            cost_printer = util.CostPrinter(util.print_cost,
                                            {'master': True, 'log_format': 'load model cost %s sec',
                                             'stdout': stdout_str})
            self.print_log("going to load model %s" % self._train_pass._checkpoint_model_path, {'master': True})
            # if config.need_reqi_changeslot and config.reqi_dnn_plugin_day >= self._train_pass.date()
            #    and config.reqi_dnn_plugin_pass >= self._pass_id:
            #    fleet.load_one_table(0, self._train_pass._checkpoint_model_path)
            # else:
            fleet.init_server(self._train_pass._checkpoint_model_path, mode=0)
            cost_printer.done()
        if self.global_config['save_first_base']:
            self.print_log("save_first_base=True", {'master': True})
            self.print_log("going to save xbox base model", {'master': True, 'stdout': stdout_str})
            self._train_pass._base_key = int(time.time())
            stdout_str += self.save_xbox_model(self._train_pass.date(), 0, self._train_pass._base_key, "")
        context['status'] = 'begin_day'

    def begin_day(self, context):
        """R
        """
        stdout_str = ""
        if not self._train_pass.next():
            context['is_exit'] = True
        day = self._train_pass.date()
        pass_id = self._train_pass._pass_id
        self.print_log("======== BEGIN DAY:%s ========" % day, {'master': True, 'stdout': stdout_str})
        if pass_id == self._train_pass.max_pass_num_day():
            context['status'] = 'end_day'
        else:
            context['status'] = 'train_pass'

    def end_day(self, context):
        """R
        """
        day = self._train_pass.date()
        pass_id = self._train_pass._pass_id
        xbox_base_key = int(time.time())
        context['status'] = 'begin_day'

        util.rank0_print("shrink table")
        cost_printer = util.CostPrinter(util.print_cost,
                                        {'master': True, 'log_format': 'shrink table done, cost %s sec'})
        fleet.shrink_sparse_table()
        for executor in self._exector_context:
            self._exector_context[executor]['model'].shrink({
                'scope': self._exector_context[executor]['scope'],
                'decay': self.global_config['optimizer']['dense_decay_rate']
            })
        cost_printer.done()

        next_date = self._train_pass.date(delta_day=1)
        util.rank0_print("going to save xbox base model")
        self.save_xbox_model(next_date, 0, xbox_base_key, "")
        util.rank0_print("going to save batch model")
        self.save_model(next_date, 0, xbox_base_key)
        self._train_pass._base_key = xbox_base_key
        fleet._role_maker._barrier_worker()

    def train_pass(self, context):
        """R
        """
        stdout_str = ""
        day = self._train_pass.date()
        pass_id = self._train_pass._pass_id
        base_key = self._train_pass._base_key
        pass_time = self._train_pass._current_train_time.strftime("%Y%m%d%H%M")
        self.print_log("    ==== begin delta:%s ========" % pass_id, {'master': True, 'stdout': stdout_str})
        train_begin_time = time.time()

        cost_printer = util.CostPrinter(util.print_cost, \
                                        {'master': True, 'log_format': 'load into memory done, cost %s sec',
                                         'stdout': stdout_str})
        current_dataset = {}
        for name in self._dataset:
            current_dataset[name] = self._dataset[name].load_dataset({
                'node_num': fleet.worker_num(), 'node_idx': fleet.worker_index(),
                'begin_time': pass_time, 'time_window_min': self._train_pass._interval_per_pass
            })
        fleet._role_maker._barrier_worker()
        cost_printer.done()

        util.rank0_print("going to global shuffle")
        cost_printer = util.CostPrinter(util.print_cost, {
            'master': True, 'stdout': stdout_str,
            'log_format': 'global shuffle done, cost %s sec'})
        for name in current_dataset:
            current_dataset[name].global_shuffle(fleet, self.global_config['dataset']['shuffle_thread'])
        cost_printer.done()
        # str(dataset.get_shuffle_data_size(fleet))
        fleet._role_maker._barrier_worker()

        if self.global_config['prefetch_data']:
            next_pass_time = (self._train_pass._current_train_time +
                              datetime.timedelta(minutes=self._train_pass._interval_per_pass)).strftime("%Y%m%d%H%M")
            for name in self._dataset:
                self._dataset[name].preload_dataset({
                    'node_num': fleet.worker_num(), 'node_idx': fleet.worker_index(),
                    'begin_time': next_pass_time, 'time_window_min': self._train_pass._interval_per_pass
                })

        fleet._role_maker._barrier_worker()
        pure_train_begin = time.time()
        for executor in self.global_config['executor']:
            self.run_executor(executor, current_dataset[executor['dataset_name']], stdout_str)
        cost_printer = util.CostPrinter(util.print_cost, \
                                        {'master': True, 'log_format': 'release_memory cost %s sec'})
        for name in current_dataset:
            current_dataset[name].release_memory()
        pure_train_cost = time.time() - pure_train_begin

        if self._train_pass.is_checkpoint_pass(pass_id):
            self.save_model(day, pass_id, base_key)

        train_end_time = time.time()
        train_cost = train_end_time - train_begin_time
        other_cost = train_cost - pure_train_cost
        log_str = "finished train day %s pass %s time cost:%s sec job time cost:" % (day, pass_id, train_cost)
        for executor in self._exector_context:
            log_str += '[' + executor + ':' + str(self._exector_context[executor]['cost']) + ']'
        log_str += '[other_cost:' + str(other_cost) + ']'
        util.rank0_print(log_str)
        stdout_str += util.now_time_str() + log_str
        sys.stdout.write(stdout_str)
        fleet._role_maker._barrier_worker()
        stdout_str = ""
        if pass_id == self._train_pass.max_pass_num_day():
            context['status'] = 'end_day'
            return
        elif not self._train_pass.next():
            context['is_exit'] = True
