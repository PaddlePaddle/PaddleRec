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

import os
import subprocess

import argparse
import tempfile
import yaml

from paddlerec.core.factory import TrainerFactory
from paddlerec.core.utils import envs
from paddlerec.core.utils import util

engines = {}
device = ["CPU", "GPU"]
clusters = ["SINGLE", "LOCAL_CLUSTER", "CLUSTER"]
engine_choices = ["SINGLE", "LOCAL_CLUSTER", "CLUSTER",
                  "TDM_SINGLE", "TDM_LOCAL_CLUSTER", "TDM_CLUSTER"]
custom_model = ['TDM']
model_name = ""


def engine_registry():
    engines["TRANSPILER"] = {}
    engines["PSLIB"] = {}

    engines["TRANSPILER"]["SINGLE"] = single_engine
    engines["TRANSPILER"]["LOCAL_CLUSTER"] = local_cluster_engine
    engines["TRANSPILER"]["CLUSTER"] = cluster_engine

    engines["PSLIB"]["SINGLE"] = local_mpi_engine
    engines["PSLIB"]["LOCAL_CLUSTER"] = local_mpi_engine
    engines["PSLIB"]["CLUSTER"] = cluster_mpi_engine


def get_inters_from_yaml(file, filter):
    with open(file, 'r') as rb:
        _envs = yaml.load(rb.read(), Loader=yaml.FullLoader)

    flattens = envs.flatten_environs(_envs)

    inters = {}
    for k, v in flattens.items():
        if k.startswith(filter):
            inters[k] = v
    return inters


def get_engine(args):
    transpiler = get_transpiler()
    run_extras = get_inters_from_yaml(args.model, "train.")

    engine = run_extras.get("train.engine", "single")
    engine = engine.upper()

    if engine not in engine_choices:
        raise ValueError("train.engin can not be chosen in {}".format(engine_choices))

    print("engines: \n{}".format(engines))

    run_engine = engines[transpiler].get(engine, None)

    return run_engine


def get_transpiler():
    FNULL = open(os.devnull, 'w')
    cmd = ["python", "-c",
           "import paddle.fluid as fluid; fleet_ptr = fluid.core.Fleet(); [fleet_ptr.copy_table_by_feasign(10, 10, [2020, 1010])];"]
    proc = subprocess.Popen(cmd, stdout=FNULL, stderr=FNULL, cwd=os.getcwd())
    ret = proc.wait()
    if ret == -11:
        return "PSLIB"
    else:
        return "TRANSPILER"


def set_runtime_envs(cluster_envs, engine_yaml):
    if cluster_envs is None:
        cluster_envs = {}

    engine_extras = get_inters_from_yaml(engine_yaml, "train.trainer.")
    if "train.trainer.threads" in engine_extras and "CPU_NUM" in cluster_envs:
        cluster_envs["CPU_NUM"] = engine_extras["train.trainer.threads"]

    envs.set_runtime_environs(cluster_envs)
    envs.set_runtime_environs(engine_extras)

    need_print = {}
    for k, v in os.environ.items():
        if k.startswith("train.trainer."):
            need_print[k] = v

    print(envs.pretty_print_envs(need_print, ("Runtime Envs", "Value")))


def get_trainer_prefix(args):
    if model_name in custom_model:
        return model_name.upper()
    return ""


def single_engine(args):
    trainer = get_trainer_prefix(args) + "SingleTrainer"
    single_envs = {}
    single_envs["train.trainer.trainer"] = trainer
    single_envs["train.trainer.threads"] = "2"
    single_envs["train.trainer.engine"] = "single"
    single_envs["train.trainer.platform"] = envs.get_platform()
    print("use {} engine to run model: {}".format(trainer, args.model))

    set_runtime_envs(single_envs, args.model)
    trainer = TrainerFactory.create(args.model)
    return trainer


def cluster_engine(args):
    def update_workspace(cluster_envs):
        workspace = cluster_envs.get("engine_workspace", None)

        if not workspace:
            return
        path = envs.path_adapter(workspace)
        for name, value in cluster_envs.items():
            if isinstance(value, str):
                value = value.replace("{workspace}", path)
                value = envs.windows_path_converter(value)
                cluster_envs[name] = value

    def master():
        role = "MASTER"
        from paddlerec.core.engine.cluster.cluster import ClusterEngine
        with open(args.backend, 'r') as rb:
            _envs = yaml.load(rb.read(), Loader=yaml.FullLoader)

        flattens = envs.flatten_environs(_envs, "_")
        flattens["engine_role"] = role
        flattens["engine_run_config"] = args.model
        flattens["engine_temp_path"] = tempfile.mkdtemp()
        update_workspace(flattens)

        envs.set_runtime_environs(flattens)
        print(envs.pretty_print_envs(flattens, ("Submit Runtime Envs", "Value")))

        launch = ClusterEngine(None, args.model)
        return launch

    def worker():
        role = "WORKER"
        trainer = get_trainer_prefix(args) + "ClusterTrainer"
        cluster_envs = {}
        cluster_envs["train.trainer.trainer"] = trainer
        cluster_envs["train.trainer.engine"] = "cluster"
        cluster_envs["train.trainer.threads"] = envs.get_runtime_environ("CPU_NUM")
        cluster_envs["train.trainer.platform"] = envs.get_platform()
        print("launch {} engine with cluster to with model: {}".format(
            trainer, args.model))
        set_runtime_envs(cluster_envs, args.model)

        trainer = TrainerFactory.create(args.model)
        return trainer

    role = os.getenv("PADDLE_PADDLEREC_ROLE", "MASTER")

    if role == "WORKER":
        return worker()
    else:
        return master()


def cluster_mpi_engine(args):
    print("launch cluster engine with cluster to run model: {}".format(args.model))

    cluster_envs = {}
    cluster_envs["train.trainer.trainer"] = "CtrCodingTrainer"
    cluster_envs["train.trainer.platform"] = envs.get_platform()

    set_runtime_envs(cluster_envs, args.model)

    trainer = TrainerFactory.create(args.model)
    return trainer


def local_cluster_engine(args):
    from paddlerec.core.engine.local_cluster import LocalClusterEngine

    trainer = get_trainer_prefix(args) + "ClusterTrainer"
    cluster_envs = {}
    cluster_envs["server_num"] = 1
    cluster_envs["worker_num"] = 1
    cluster_envs["start_port"] = envs.find_free_port()
    cluster_envs["log_dir"] = "logs"
    cluster_envs["train.trainer.trainer"] = trainer
    cluster_envs["train.trainer.strategy"] = "async"
    cluster_envs["train.trainer.threads"] = "2"
    cluster_envs["train.trainer.engine"] = "local_cluster"
    cluster_envs["train.trainer.platform"] = envs.get_platform()

    cluster_envs["CPU_NUM"] = "2"
    print("launch {} engine with cluster to run model: {}".format(trainer, args.model))

    set_runtime_envs(cluster_envs, args.model)
    launch = LocalClusterEngine(cluster_envs, args.model)
    return launch


def local_mpi_engine(args):
    print("launch cluster engine with cluster to run model: {}".format(args.model))
    from paddlerec.core.engine.local_mpi import LocalMPIEngine

    print("use 1X1 MPI ClusterTraining at localhost to run model: {}".format(args.model))

    mpi = util.run_which("mpirun")
    if not mpi:
        raise RuntimeError("can not find mpirun, please check environment")
    cluster_envs = {}
    cluster_envs["mpirun"] = mpi
    cluster_envs["train.trainer.trainer"] = "CtrCodingTrainer"
    cluster_envs["log_dir"] = "logs"
    cluster_envs["train.trainer.engine"] = "local_cluster"

    cluster_envs["train.trainer.platform"] = envs.get_platform()

    set_runtime_envs(cluster_envs, args.model)
    launch = LocalMPIEngine(cluster_envs, args.model)
    return launch


def get_abs_model(model):
    if model.startswith("paddlerec."):
        dir = envs.path_adapter(model)
        path = os.path.join(dir, "config.yaml")
    else:
        if not os.path.isfile(model):
            raise IOError("model config: {} invalid".format(model))
        path = model
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--backend", type=str, default=None)

    abs_dir = os.path.dirname(os.path.abspath(__file__))
    envs.set_runtime_environs({"PACKAGE_BASE": abs_dir})

    args = parser.parse_args()

    model_name = args.model.split('.')[-1]
    args.model = get_abs_model(args.model)
    engine_registry()

    which_engine = get_engine(args)
    engine = which_engine(args)
    engine.run()
