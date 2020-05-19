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
from __future__ import unicode_literals

import copy
import os
import sys
import subprocess

from paddlerec.core.engine.engine import Engine


class LocalMPIEngine(Engine):
    def start_procs(self):
        logs_dir = self.envs["log_dir"]

        default_env = os.environ.copy()
        current_env = copy.copy(default_env)
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)
        procs = []
        log_fns = []

        factory = "paddlerec.core.factory"
        cmd = "mpirun -npernode 2 -timestamp-output -tag-output".split(" ")
        cmd.extend([sys.executable, "-u", "-m", factory, self.trainer])

        if logs_dir is not None:
            os.system("mkdir -p {}".format(logs_dir))
            fn = open("%s/job.log" % logs_dir, "w")
            log_fns.append(fn)
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn, cwd=os.getcwd())
        else:
            proc = subprocess.Popen(cmd, env=current_env, cwd=os.getcwd())
        procs.append(proc)

        for i in range(len(procs)):
            if len(log_fns) > 0:
                log_fns[i].close()
            procs[i].wait()
        print("all workers and parameter servers already completed", file=sys.stderr)

    def run(self):
        self.start_procs()
