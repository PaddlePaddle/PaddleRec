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

import datetime
import os
import sys
import time
import numpy as np
from paddle import fluid

from paddlerec.core.utils import fs as fs


def save_program_proto(path, program=None):

    if program is None:
        _program = fluid.default_main_program()
    else:
        _program = program

    with open(path, "wb") as f:
        f.write(_program.desc.serialize_to_string())


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def run_which(command):
    regex = "/usr/bin/which: no {} in"
    ret = run_shell_cmd("which {}".format(command))
    if ret.startswith(regex.format(command)):
        return None
    else:
        return ret


def run_shell_cmd(command):
    assert command is not None and isinstance(command, str)
    return os.popen(command).read().strip()


def get_env_value(env_name):
    """
    get os environment value
    """
    return os.popen("echo -n ${" + env_name + "}").read().strip()


def now_time_str():
    """
    get current format str_time
    """
    return "\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "[0]:"


def get_absolute_path(path, params):
    """R
    """
    if path.startswith('afs:') or path.startswith('hdfs:'):
        sub_path = path.split('fs:')[1]
        if ':' in sub_path:  # such as afs://xxx:prot/xxxx
            return path
        elif 'fs_name' in params:
            return params['fs_name'] + sub_path
    else:
        return path


def make_datetime(date_str, fmt=None):
    """
    create a datetime instance by date_string
    Args:
        date_str: such as 2020-01-14
        date_str_format: "%Y-%m-%d"
    Return:
        datetime 
    """
    if fmt is None:
        if len(date_str) == 8:  # %Y%m%d
            return datetime.datetime.strptime(date_str, '%Y%m%d')
        if len(date_str) == 12:  # %Y%m%d%H%M
            return datetime.datetime.strptime(date_str, '%Y%m%d%H%M')
    return datetime.datetime.strptime(date_str, fmt)


def wroker_numric_opt(fleet, value, env, opt):
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


def worker_numric_sum(fleet, value, env="mpi"):
    """R
    """
    return wroker_numric_opt(fleet, value, env, "sum")


def worker_numric_avg(fleet, value, env="mpi"):
    """R
    """
    return worker_numric_sum(fleet, value, env) / fleet.worker_num()


def worker_numric_min(fleet, value, env="mpi"):
    """R
    """
    return wroker_numric_opt(fleet, value, env, "min")


def worker_numric_max(fleet, value, env="mpi"):
    """R
    """
    return wroker_numric_opt(fleet, value, env, "max")


def print_log(log_str, params):
    """R
    """
    time_str = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
    log_str = time_str + " " + log_str
    if 'master' in params and params['master']:
        if 'index' in params and params['index'] == 0:
            print(log_str)
    else:
        print(log_str)
    sys.stdout.flush()
    if 'stdout' in params:
        params['stdout'] += log_str + '\n'


def rank0_print(log_str, fleet):
    """R
    """
    print_log(log_str, {'master': True, 'index': fleet.worker_index()})


def print_cost(cost, params):
    """R
    """
    log_str = params['log_format'] % cost
    print_log(log_str, params)
    return log_str


class CostPrinter(object):
    """
    For count cost time && print cost log
    """

    def __init__(self, callback, callback_params):
        """R
        """
        self.reset(callback, callback_params)
        pass

    def __del__(self):
        """R
        """
        if not self._done:
            self.done()
        pass

    def reset(self, callback, callback_params):
        """R
        """
        self._done = False
        self._callback = callback
        self._callback_params = callback_params
        self._begin_time = time.time()
        pass

    def done(self):
        """R
        """
        cost = time.time() - self._begin_time
        log_str = self._callback(cost, self._callback_params)  # cost(s)
        self._done = True
        return cost, log_str


class PathGenerator(object):
    """
    generate path with template & runtime variables
    """

    def __init__(self, config):
        """R
        """
        self._templates = {}
        self.add_path_template(config)
        pass

    def add_path_template(self, config):
        """R
        """
        if 'templates' in config:
            for template in config['templates']:
                self._templates[template['name']] = template['template']
        pass

    def generate_path(self, template_name, param):
        """R
        """
        if template_name in self._templates:
            if 'time_format' in param:
                str = param['time_format'].strftime(self._templates[
                    template_name])
                return str.format(**param)
            return self._templates[template_name].format(**param)
        else:
            return ""
