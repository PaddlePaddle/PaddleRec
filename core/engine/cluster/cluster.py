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
import warnings

from paddlerec.core.engine.engine import Engine
from paddlerec.core.factory import TrainerFactory
from paddlerec.core.utils import envs


class ClusterEngine(Engine):
    def __init_impl__(self):
        self.role = envs.get_runtime_environ("engine_role")
        if self.role == "WORKER":
            return

        abs_dir = os.path.dirname(os.path.abspath(__file__))
        os.environ["abs_dir"] = str(abs_dir)

        self.backend = envs.get_runtime_environ("backend")
        if not self.backend:
            self.backend = ""
        self.backend = self.backend.upper()
        if self.backend == "PADDLECLOUD":
            self.submit_script = os.path.join(abs_dir, "cloud/cluster.sh")
        elif self.backend == "KUBERNETES":
            self.submit_script = os.path.join(abs_dir, "k8s/cluster.sh")
        else:
            raise ValueError("{} can not be supported now".format(
                self.backend))

    def start_worker_procs(self):
        trainer = TrainerFactory.create(self.trainer)
        trainer.run()

    def start_master_procs(self):
        if self.backend == "PADDLECLOUD":
            self.paddlecloud_env_check()
        elif self.backend == "KUBERNETES":
            self.kubernetes_env_check()

        default_env = os.environ.copy()
        current_env = copy.copy(default_env)
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)

        cmd = ("bash {}".format(self.submit_script)).split(" ")
        proc = subprocess.Popen(cmd, env=current_env, cwd=os.getcwd())
        proc.wait()

    @staticmethod
    def workspace_replace():
        workspace = envs.get_runtime_environ("workspace")

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
                    "Paddlecloud with Mpi don't support GPU training, check your config"
                )
        elif cluster_type.upper() == "K8S":
            if fleet_mode == "PS":
                if device == "CPU":
                    raise ValueError(
                        "PS-CPU on paddlecloud is not supported at this time, comming soon"
                    )
                elif device == "GPU":
                    raise ValueError(
                        "PS-GPU on paddlecloud is not supported at this time, comming soon"
                    )
            if fleet_mode == "COLLECTIVE":
                if device == "GPU":
                    cluster_env_check_tool = CloudCollectiveEnv()
                elif device == "CPU":
                    raise ValueError(
                        "Unexpected config -> device: CPU with fleet_mode: Collective, check your config"
                    )
        else:
            raise ValueError("cluster_type {} error, must in MPI/K8S".format(
                cluster_type))

        cluster_env_check_tool.env_check()
        cluster_env_check_tool.env_set()

    def kubernetes_env_check(self):
        pass


class ClusterEnvBase(object):
    def __init__(self):
        # get backend env
        backend_yaml = envs.get_runtime_environ("backend_yaml")
        _env = envs.load_yaml(backend_yaml)
        self.backend_env = envs.flatten_environs(_env, ".")
        self.cluster_env = {}

    def env_check(self):
        # check common env
        # fs_name & fs_ugi
        self.cluster_env["FS_NAME"] = self.backend_env.get("config.fs_name",
                                                           "")
        self.cluster_env["FS_UGI"] = self.backend_env.get("config.fs_ugi", "")
        if self.cluster_env["FS_NAME"] == "" or self.cluster_env[
                "FS_UGI"] == "":
            raise ValueError(
                "No -- FS_UGI or FS_NAME -- found in your backend.yaml, please check."
            )

        # output_path
        self.cluster_env["OUTPUT_PATH"] = self.backend_env.get(
            "config.output_path", "")
        if self.cluster_env["OUTPUT_PATH"] == "":
            warnings.warn(
                "Job output_path not set! Please check your backend yaml.",
                category=UserWarning,
                stacklevel=2)

        # paddle_version
        self.cluster_env["PADDLE_VERSION"] = self.backend_env.get(
            "config.paddle_version", "1.7.2")

        # communicator
        self.cluster_env[
            "FLAGS_communicator_is_sgd_optimizer"] = self.backend_env.get(
                "config.communicator.FLAGS_communicator_is_sgd_optimizer", 0)
        self.cluster_env[
            "FLAGS_communicator_send_queue_size"] = self.backend_env.get(
                "config.communicator.FLAGS_communicator_send_queue_size", 5)
        self.cluster_env[
            "FLAGS_communicator_thread_pool_size"] = self.backend_env.get(
                "config.communicator.FLAGS_communicator_thread_pool_size", 32)
        self.cluster_env[
            "FLAGS_communicator_max_merge_var_num"] = self.backend_env.get(
                "config.communicator.FLAGS_communicator_max_merge_var_num", 5)
        self.cluster_env[
            "FLAGS_communicator_max_send_grad_num_before_recv"] = self.backend_env.get(
                "config.communicator.FLAGS_communicator_max_send_grad_num_before_recv",
                5)
        self.cluster_env["FLAGS_communicator_fake_rpc"] = self.backend_env.get(
            "config.communicator.FLAGS_communicator_fake_rpc", 0)
        self.cluster_env["FLAGS_rpc_retry_times"] = self.backend_env.get(
            "config.communicator.FLAGS_rpc_retry_times", 3)

        # ak & sk
        self.cluster_env["AK"] = self.backend_env.get("submit.ak", "")
        self.cluster_env["SK"] = self.backend_env.get("submit.sk", "")
        if self.cluster_env["AK"] == "" or self.cluster_env["SK"] == "":
            raise ValueError(
                "No -- AK or SK -- found in your backend.yaml, please check.")

        # priority
        self.cluster_env["PRIORITY"] = self.backend_env.get("submit.priority",
                                                            "high")

        # job name
        self.cluster_env["JOB_NAME"] = self.backend_env.get(
            "submit.job_name", "PaddleRecClusterJob")

        # group
        self.cluster_env["GROUP_NAME"] = self.backend_env.get("submit.group",
                                                              "paddle")

        # start_cmd
        self.cluster_env["START_CMD"] = self.backend_env.get(
            "submit.start_cmd", "python -m paddlerec.run -m config.yaml")

        # files
        self.cluster_env["FILES"] = self.backend_env.get("submit.files", "")
        if self.cluster_env["FILES"] == "":
            raise ValueError(
                "No -- files -- found in your backend.yaml, please check.")

    def env_set(self):
        envs.set_runtime_environs(self.cluster_env)
        flattens = envs.flatten_environs(self.cluster_env)
        print(envs.pretty_print_envs(flattens, ("Cluster Envs", "Value")))


class PaddleCloudMpiEnv(ClusterEnvBase):
    def __init__(self):
        super(PaddleCloudMpiEnv, self).__init__()

    def env_check(self):
        super(PaddleCloudMpiEnv, self).env_check()

        # check mpi env

        self.cluster_env["DISTRIBUTE_MODE"] = "PS_CPU_MPI"

        # train_data_path
        self.cluster_env["TRAIN_DATA_PATH"] = self.backend_env.get(
            "config.train_data_path", "")
        if self.cluster_env["TRAIN_DATA_PATH"] == "":
            raise ValueError(
                "No -- TRAIN_DATA_PATH -- found in your backend.yaml, please check."
            )
        # test_data_path
        self.cluster_env["TEST_DATA_PATH"] = self.backend_env.get(
            "config.test_data_path", "")
        if self.cluster_env["TEST_DATA_PATH"] == "":
            warnings.warn(
                "Job test_data_path not set! Please check your backend yaml.",
                category=UserWarning,
                stacklevel=2)

        # thirdparty_path
        self.cluster_env["THIRDPARTY_PATH"] = self.backend_env.get(
            "config.thirdparty_path", "")
        if self.cluster_env["THIRDPARTY_PATH"] == "":
            warnings.warn(
                "Job thirdparty_path not set! Please check your backend yaml.",
                category=UserWarning,
                stacklevel=2)

        # nodes
        self.cluster_env["MPI_NODES"] = self.backend_env.get("submit.nodes", 1)


class PaddleCloudK8sEnv(ClusterEnvBase):
    def __init__(self):
        super(PaddleCloudK8sEnv, self).__init__()

    def env_check(self):
        super(PaddleCloudK8sEnv, self).env_check()

        # check afs_remote_mount_point
        self.cluster_env["AFS_REMOTE_MOUNT_POINT"] = self.backend_env.get(
            "config.afs_remote_mount_point", "")
        if self.cluster_env["AFS_REMOTE_MOUNT_POINT"] == "":
            warnings.warn(
                "Job afs_remote_mount_point not set! Please check your backend yaml.",
                category=UserWarning,
                stacklevel=2)
        warnings.warn(
            "The remote mount point will be mounted to the ./afs/",
            category=UserWarning,
            stacklevel=2)


class CloudCollectiveEnv(PaddleCloudK8sEnv):
    def __init__(self):
        super(CloudCollectiveEnv, self).__init__()

    def env_check(self):
        super(CloudCollectiveEnv, self).env_check()

        self.cluster_env["DISTRIBUTE_MODE"] = "COLLECTIVE_GPU_K8S"
        self.cluster_env["K8S_TRAINERS"] = self.backend_env.get(
            "submit.k8s_trainers", 1)
        self.cluster_env["K8S_GPU_CARD"] = self.backend_env.get(
            "submit.k8s_gpu_card", 1)
        self.cluster_env["K8S_CPU_CORES"] = self.backend_env.get(
            "submit.k8s_cpu_cores", 1)
