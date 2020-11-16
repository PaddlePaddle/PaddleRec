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
import logging
import tempfile
import shutil

from paddlerec.core.engine.engine import Engine
from paddlerec.core.utils import envs
import paddlerec.core.engine.cluster_utils as cluster_utils

logging.basicConfig(
    format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class LocalClusterEngine(Engine):
    def start_procs(self):
        fleet_mode = self.envs["fleet_mode"]
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

        self.gloo_rendezvous_dir = tempfile.mkdtemp()
        gloo_http_port = str(envs.find_free_port())
        self.gloo_endpoints = ":".join(["127.0.0.1", gloo_http_port])

        if fleet_mode.upper() == "PS":
            for i in range(server_num - 1):
                while True:
                    new_port = envs.find_free_port()
                    if new_port not in ports:
                        ports.append(new_port)
                        break
            user_endpoints = ",".join(["127.0.0.1:" + str(x) for x in ports])

            user_endpoints_ips = [
                x.split(":")[0] for x in user_endpoints.split(",")
            ]
            user_endpoints_port = [
                x.split(":")[1] for x in user_endpoints.split(",")
            ]

            factory = "paddlerec.core.factory"
            cmd = [sys.executable, "-u", "-m", factory, self.trainer]

            for i in range(server_num):
                current_env.update({
                    "PADDLE_PSERVERS_IP_PORT_LIST": user_endpoints,
                    "PADDLE_PORT": user_endpoints_port[i],
                    "TRAINING_ROLE": "PSERVER",
                    "PADDLE_TRAINERS_NUM": str(worker_num),
                    "POD_IP": user_endpoints_ips[i],
                    "PADDLE_WITH_GLOO": "1",
                    "PADDLE_GLOO_RENDEZVOUS": "3",
                    "PADDLE_GLOO_FS_PATH": self.gloo_rendezvous_dir,
                    "PADDLE_GLOO_HTTP_ENDPOINT": self.gloo_endpoints
                })

                os.system("mkdir -p {}".format(logs_dir))
                fn = open("%s/server.%d" % (logs_dir, i), "w")
                log_fns.append(fn)
                proc = subprocess.Popen(
                    cmd,
                    env=current_env,
                    stdout=fn,
                    stderr=fn,
                    cwd=os.getcwd())
                procs.append(proc)

            for i in range(worker_num):
                current_env.update({
                    "PADDLE_PSERVERS_IP_PORT_LIST": user_endpoints,
                    "PADDLE_TRAINERS_NUM": str(worker_num),
                    "TRAINING_ROLE": "TRAINER",
                    "PADDLE_TRAINER_ID": str(i),
                    "PADDLE_WITH_GLOO": "1",
                    "PADDLE_GLOO_RENDEZVOUS": "3",
                    "PADDLE_GLOO_FS_PATH": self.gloo_rendezvous_dir,
                    "PADDLE_GLOO_HTTP_ENDPOINT": self.gloo_endpoints
                })

                os.system("mkdir -p {}".format(logs_dir))
                fn = open("%s/worker.%d" % (logs_dir, i), "w")
                log_fns.append(fn)
                proc = subprocess.Popen(
                    cmd,
                    env=current_env,
                    stdout=fn,
                    stderr=fn,
                    cwd=os.getcwd())
                procs.append(proc)

        elif fleet_mode.upper() == "COLLECTIVE":
            cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices is None or cuda_visible_devices == "":
                selected_gpus = [
                    x.strip() for x in self.envs["selected_gpus"].split(",")
                ]
            else:
                # change selected_gpus into relative values
                # e.g. CUDA_VISIBLE_DEVICES=4,5,6,7; args.selected_gpus=4,5,6,7;
                # therefore selected_gpus=0,1,2,3
                cuda_visible_devices_list = cuda_visible_devices.split(',')
                for x in self.envs["selected_gpus"].split(","):
                    assert x in cuda_visible_devices_list, "Can't find "\
                        "your selected_gpus %s in CUDA_VISIBLE_DEVICES[%s]."\
                        % (x, cuda_visible_devices)
                selected_gpus = [
                    cuda_visible_devices_list.index(x.strip())
                    for x in self.envs["selected_gpus"].split(",")
                ]
            selected_gpus_num = len(selected_gpus)

            factory = "paddlerec.core.factory"
            cmd = [sys.executable, "-u", "-m", factory, self.trainer]

            print("use_paddlecloud_flag:{}".format(
                cluster_utils.use_paddlecloud()))
            if cluster_utils.use_paddlecloud():
                cluster, pod = cluster_utils.get_cloud_cluster(selected_gpus)
                logger.info("get cluster from cloud:{}".format(cluster))
                procs = cluster_utils.start_local_trainers(
                    cluster, pod, cmd, log_dir=logs_dir)

            else:
                # trainers_num = 1 or not use paddlecloud ips="a,b"
                for i in range(selected_gpus_num - 1):
                    while True:
                        new_port = envs.find_free_port()
                        if new_port not in ports:
                            ports.append(new_port)
                            break
                user_endpoints = ",".join(
                    ["127.0.0.1:" + str(x) for x in ports])
                for i in range(selected_gpus_num):
                    current_env.update({
                        "PADDLE_TRAINER_ENDPOINTS": user_endpoints,
                        "PADDLE_CURRENT_ENDPOINTS": user_endpoints[i],
                        "PADDLE_TRAINERS_NUM": str(worker_num),
                        "TRAINING_ROLE": "TRAINER",
                        "PADDLE_TRAINER_ID": str(i),
                        "FLAGS_selected_gpus": str(selected_gpus[i]),
                        "PADDLEREC_GPU_NUMS": str(selected_gpus_num),
                        "PADDLE_WITH_GLOO": "1",
                        "PADDLE_GLOO_RENDEZVOUS": "3",
                        "PADDLE_GLOO_FS_PATH": self.gloo_rendezvous_dir,
                    })

                    os.system("mkdir -p {}".format(logs_dir))
                    fn = open("%s/worker.%d" % (logs_dir, i), "w")
                    log_fns.append(fn)
                    proc = subprocess.Popen(
                        cmd,
                        env=current_env,
                        stdout=fn,
                        stderr=fn,
                        cwd=os.getcwd())
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
        print(
            "all workers already completed, you can view logs under the `{}` directory".
            format(logs_dir),
            file=sys.stderr)
        if os.path.exists(self.gloo_rendezvous_dir):
            shutil.rmtree(self.gloo_rendezvous_dir)

    def run(self):
        self.start_procs()
