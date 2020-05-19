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
from paddlerec.core.utils import envs


class LocalClusterEngine(Engine):
    def start_procs(self):
        worker_num = self.envs["worker_num"]
        server_num = self.envs["server_num"]
        ports = [self.envs["start_port"]]
        logs_dir = self.envs["log_dir"]

        default_env = os.environ.copy()
        current_env = copy.copy(default_env)
        current_env["CLUSTER_INSTANCE"] = "1"
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)
        procs = []
        log_fns = []

        for i in range(server_num - 1):
            while True:
                new_port = envs.find_free_port()
                if new_port not in ports:
                    ports.append(new_port)
                    break
        user_endpoints = ",".join(["127.0.0.1:" + str(x) for x in ports])
        user_endpoints_ips = [x.split(":")[0]
                              for x in user_endpoints.split(",")]
        user_endpoints_port = [x.split(":")[1]
                               for x in user_endpoints.split(",")]

        factory = "paddlerec.core.factory"
        cmd = [sys.executable, "-u", "-m", factory, self.trainer]

        for i in range(server_num):
            current_env.update({
                "PADDLE_PSERVERS_IP_PORT_LIST": user_endpoints,
                "PADDLE_PORT": user_endpoints_port[i],
                "TRAINING_ROLE": "PSERVER",
                "PADDLE_TRAINERS_NUM": str(worker_num),
                "POD_IP": user_endpoints_ips[i]
            })

            os.system("mkdir -p {}".format(logs_dir))
            fn = open("%s/server.%d" % (logs_dir, i), "w")
            log_fns.append(fn)
            proc = subprocess.Popen(
                cmd, env=current_env, stdout=fn, stderr=fn, cwd=os.getcwd())
            procs.append(proc)

        for i in range(worker_num):
            current_env.update({
                "PADDLE_PSERVERS_IP_PORT_LIST": user_endpoints,
                "PADDLE_TRAINERS_NUM": str(worker_num),
                "TRAINING_ROLE": "TRAINER",
                "PADDLE_TRAINER_ID": str(i)
            })

            os.system("mkdir -p {}".format(logs_dir))
            fn = open("%s/worker.%d" % (logs_dir, i), "w")
            log_fns.append(fn)
            proc = subprocess.Popen(
                cmd, env=current_env, stdout=fn, stderr=fn, cwd=os.getcwd())
            procs.append(proc)

        # only wait worker to finish here
        for i, proc in enumerate(procs):
            if i < server_num:
                continue
            procs[i].wait()
            if len(log_fns) > 0:
                log_fns[i].close()

        for i in range(server_num):
            if len(log_fns) > 0:
                log_fns[i].close()
            procs[i].terminate()
        print("all workers already completed, you can view logs under the `{}` directory".format(logs_dir),
              file=sys.stderr)

    def run(self):
        self.start_procs()
