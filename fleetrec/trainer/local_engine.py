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


def start_procs(args, yaml):
    worker_num = args["worker_num"]
    server_num = args["server_num"]
    start_port = args["start_port"]
    logs_dir = args["log_dir"]

    default_env = os.environ.copy()
    current_env = copy.copy(default_env)
    current_env["CLUSTER_INSTANCE"] = "1"
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)
    procs = []
    log_fns = []
    ports = range(start_port, start_port + server_num, 1)
    user_endpoints = ",".join(["127.0.0.1:" + str(x) for x in ports])
    user_endpoints_ips = [x.split(":")[0] for x in user_endpoints.split(",")]
    user_endpoints_port = [x.split(":")[1] for x in user_endpoints.split(",")]

    factory = "fleetrec.trainer.factory"
    cmd = [sys.executable, "-u", "-m", factory, yaml]

    for i in range(server_num):
        current_env.update({
            "PADDLE_PSERVERS_IP_PORT_LIST": user_endpoints,
            "PADDLE_PORT": user_endpoints_port[i],
            "TRAINING_ROLE": "PSERVER",
            "PADDLE_TRAINERS_NUM": str(worker_num),
            "POD_IP": user_endpoints_ips[i]
        })

        if logs_dir is not None:
            os.system("mkdir -p {}".format(logs_dir))
            fn = open("%s/server.%d" % (logs_dir, i), "w")
            log_fns.append(fn)
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn, cwd=os.getcwd())
        else:
            proc = subprocess.Popen(cmd, env=current_env, cwd=os.getcwd())
        procs.append(proc)

    for i in range(worker_num):
        current_env.update({
            "PADDLE_PSERVERS_IP_PORT_LIST": user_endpoints,
            "PADDLE_TRAINERS_NUM": str(worker_num),
            "TRAINING_ROLE": "TRAINER",
            "PADDLE_TRAINER_ID": str(i)
        })

        if logs_dir is not None:
            os.system("mkdir -p {}".format(logs_dir))
            fn = open("%s/worker.%d" % (logs_dir, i), "w")
            log_fns.append(fn)
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn, cwd=os.getcwd())
        else:
            proc = subprocess.Popen(cmd, env=current_env, cwd=os.getcwd())
        procs.append(proc)

    # only wait worker to finish here
    for i, proc in enumerate(procs):
        if i < server_num:
            continue
        procs[i].wait()
        if len(log_fns) > 0:
            log_fns[i].close()

    print("all workers exit, going to finish parameter server", file=sys.stderr)
    for i in range(server_num):
        if len(log_fns) > 0:
            log_fns[i].close()
        procs[i].terminate()
    print("all parameter server are killed", file=sys.stderr)

class Launch():
    def __init__(self, envs, trainer):
        self.envs = envs
        self.trainer = trainer

    def run(self):
        start_procs(self.envs, self.trainer)

