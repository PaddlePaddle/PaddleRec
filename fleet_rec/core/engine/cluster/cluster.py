# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import unicode_literals

import subprocess
import sys
import os
import copy

from fleetrec.core.engine.engine import Engine
from fleetrec.core.utils import envs


class ClusterEngine(Engine):
    def __init_impl__(self):
        abs_dir = os.path.dirname(os.path.abspath(__file__))
        self.submit_script = os.path.join(abs_dir, "master.sh")

    def start_worker_procs(self):
        default_env = os.environ.copy()
        current_env = copy.copy(default_env)
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)

        cmd = ("bash {}".format(self.submit_script)).split(" ")
        proc = subprocess.Popen(cmd, env=current_env, cwd=os.getcwd())
        proc.wait()

        print("all workers and parameter servers already completed", file=sys.stderr)

    def start_master_procs(self):
        default_env = os.environ.copy()
        current_env = copy.copy(default_env)
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)

        cmd = ("bash {}".format(self.submit_script)).split(" ")
        proc = subprocess.Popen(cmd, env=current_env, cwd=os.getcwd())
        proc.wait()

        print("all workers and parameter servers already completed", file=sys.stderr)

    def run(self):
        role = envs.get_runtime_environ("engine_role")

        if role == "MASTER":
            self.start_master_procs()

        elif role == "WORKER":
            self.start_worker_procs()

        else:
            raise ValueError("role {} error, must in MASTER/WORKER".format(role))
