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
import subprocess

from paddlerec.core.engine.engine import Engine
from paddlerec.core.factory import TrainerFactory
from paddlerec.core.utils import envs


class ClusterEngine(Engine):
    def __init_impl__(self):
        self.role = envs.get_runtime_environ("engine_role")
        if self.role == "WORKER":
            return

        abs_dir = os.path.dirname(os.path.abspath(__file__))

        self.backend = envs.get_runtime_environ("engine_backend")
        if not self.backend:
            backend = ""
        self.backend = self.backend.upper()
        if self.backend == "PADDLECLOUD":
            self.submit_script = os.path.join(abs_dir, "cloud/cluster.sh")
        elif self.backend == "KUBERNETES":
            self.submit_script = os.path.join(abs_dir, "k8s/cluster.sh")
        else:
            raise ValueError(
                "{} can not be supported now".format(self.backend))

    def start_worker_procs(self):
        trainer = TrainerFactory.create(self.trainer)
        trainer.run()

    def start_master_procs(self):
        default_env = os.environ.copy()
        current_env = copy.copy(default_env)
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)

        if self.backend == "PADDLECLOUD":
            self.paddlecloud_env_check()
        elif self.backend == "KUBERNETES":
            self.kubernetes_env_check()

        cmd = ("bash {}".format(self.submit_script)).split(" ")
        proc = subprocess.Popen(cmd, env=current_env, cwd=os.getcwd())
        proc.wait()

    @staticmethod
    def workspace_replace():
        workspace = envs.get_runtime_environ("engine_workspace")

        for k, v in os.environ.items():
            v = v.replace("{workspace}", workspace)
            os.environ[k] = str(v)

    def run(self):
        if self.role == "MASTER":
            self.start_master_procs()

        elif self.role == "WORKER":
            self.start_worker_procs()

        else:
            raise ValueError("role {} error, must in MASTER/WORKER".format(
                self.role))

    def paddlecloud_env_check(self):
        # get fleet mode
        fleet_mode = envs.get_runtime_environ("fleet_mode")
        # get device
        device = envs.get_runtime_environ("device")
        # get cluster type
        cluster_type = envs.get_runtime_environ("cluster_type")

        cluster_env_check_tool = None
        if cluster_type.upper() == "MPI":
            if device == "CPU" and fleet_mode == "PS":
                cluster_env_check_tool = PaddleCloudMpiEnv()
            else:
                raise ValueError(
                    "Paddlecloud with Mpi don't support GPU training, check your config")
        elif cluster_type.upper() == "K8S":
            if fleet_mode == "PS":
                if device == "CPU":
                    cluster_env_check_tool = CloudPsCpuEnv()
                elif device == "GPU":
                    raise ValueError(
                        "PS-GPU is not supported at this time, comming soon")
            if fleet_mode == "COLLECTIVE":
                if device == "GPU":
                    cluster_env_check_tool = CloudCollectiveEnv()
                elif device == "CPU":
                    raise ValueError(
                        "Unexpected use-> device: CPU with fleet_mode: Collective")
        else:
            raise ValueError("cluster_type {} error, must in MPI/K8S".format(
                cluster_type))

        cluster_env_check_tool.env_check()
        cluster_env_check_tool.env_set()

    def kubernetes_env_check(self):
        pass


class ClusterEnvBase(object):
    def __init__(self):
        pass

    def env_check(self):
        pass

    def env_set(self):
        pass


class PaddleCloudMpiEnv(ClusterEnvBase):
    def env_check(self):
        pass

    def env_set(self):
        pass


class PaddleCloudK8sEnv(ClusterEnvBase):
    def env_check(self):
        pass

    def env_set(self):
        pass


class CloudPsCpuEnv(PaddleCloudK8sEnv):
    def env_check(self):
        pass

    def env_set(self):
        pass


class CloudCollectiveEnv(PaddleCloudK8sEnv):
    def env_check(self):
        pass

    def env_set(self):
        pass
