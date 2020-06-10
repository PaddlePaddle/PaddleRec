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
import sys
import argparse
import tempfile

import copy
from paddlerec.core.factory import TrainerFactory
from paddlerec.core.utils import envs
from paddlerec.core.utils import util
from paddlerec.core.utils import validation

engines = {}
device = ["CPU", "GPU"]
engine_choices = [
    "TRAIN", "SINGLE_TRAIN", "INFER", "SINGLE_INFER", "LOCAL_CLUSTER",
    "LOCAL_CLUSTER_TRAIN", "CLUSTER_TRAIN"
]


def engine_registry():
    engines["TRANSPILER"] = {}
    engines["PSLIB"] = {}

    engines["TRANSPILER"]["TRAIN"] = single_train_engine
    engines["TRANSPILER"]["SINGLE_TRAIN"] = single_train_engine
    engines["TRANSPILER"]["INFER"] = single_infer_engine
    engines["TRANSPILER"]["SINGLE_INFER"] = single_infer_engine
    engines["TRANSPILER"]["LOCAL_CLUSTER"] = local_cluster_engine
    engines["TRANSPILER"]["LOCAL_CLUSTER_TRAIN"] = local_cluster_engine
    engines["TRANSPILER"]["CLUSTER"] = cluster_engine
    engines["PSLIB"]["SINGLE_TRAIN"] = local_mpi_engine
    engines["PSLIB"]["TRAIN"] = local_mpi_engine
    engines["PSLIB"]["LOCAL_CLUSTER_TRAIN"] = local_mpi_engine
    engines["PSLIB"]["LOCAL_CLUSTER"] = local_mpi_engine
    engines["PSLIB"]["CLUSTER_TRAIN"] = cluster_mpi_engine
    engines["PSLIB"]["CLUSTER"] = cluster_mpi_engine


def get_inters_from_yaml(file, filters):
    _envs = envs.load_yaml(file)
    flattens = envs.flatten_environs(_envs)
    inters = {}
    for k, v in flattens.items():
        for f in filters:
            if k.startswith(f):
                inters[k] = v
    return inters


def get_all_inters_from_yaml(file, filters):
    _envs = envs.load_yaml(file)
    all_flattens = {}

    def fatten_env_namespace(namespace_nests, local_envs):
        for k, v in local_envs.items():
            if isinstance(v, dict):
                nests = copy.deepcopy(namespace_nests)
                nests.append(k)
                fatten_env_namespace(nests, v)
            elif (k == "dataset" or k == "phase" or
                  k == "runner") and isinstance(v, list):
                for i in v:
                    if i.get("name") is None:
                        raise ValueError("name must be in dataset list. ", v)
                    nests = copy.deepcopy(namespace_nests)
                    nests.append(k)
                    nests.append(i["name"])
                    fatten_env_namespace(nests, i)
            else:
                global_k = ".".join(namespace_nests + [k])
                all_flattens[global_k] = v

    fatten_env_namespace([], _envs)
    ret = {}
    for k, v in all_flattens.items():
        for f in filters:
            if k.startswith(f):
                ret[k] = v
    return ret


def get_modes(running_config):
    if not isinstance(running_config, dict):
        raise ValueError("get_modes arguments must be [dict]")

    modes = running_config.get("mode")
    if not modes:
        raise ValueError("yaml mast have config: mode")

    if isinstance(modes, str):
        modes = [modes]

    return modes


def get_engine(args, running_config, mode):
    transpiler = get_transpiler()
    _envs = envs.load_yaml(args.model)

    engine_class = ".".join(["runner", mode, "class"])
    engine_device = ".".join(["runner", mode, "device"])
    device_gpu_choices = ".".join(["runner", mode, "device", "selected_gpus"])

    engine = running_config.get(engine_class, None)
    if engine is None:
        raise ValueError("not find {} in yaml, please check".format(
            mode, engine_class))
    device = running_config.get(engine_device, None)

    if device is None:
        print("not find device be specified in yaml, set CPU as default")
        device = "CPU"

    if device.upper() == "GPU":
        selected_gpus = running_config.get(device_gpu_choices, None)

        if selected_gpus is None:
            print(
                "not find selected_gpus be specified in yaml, set `0` as default"
            )
            selected_gpus = ["0"]
        else:
            print("selected_gpus {} will be specified for running".format(
                selected_gpus))

        selected_gpus_num = len(selected_gpus.split(","))
        if selected_gpus_num > 1:
            engine = "LOCAL_CLUSTER"

    engine = engine.upper()
    if engine not in engine_choices:
        raise ValueError("{} can not be chosen in {}".format(engine_class,
                                                             engine_choices))

    run_engine = engines[transpiler].get(engine, None)
    return run_engine


def get_transpiler():
    FNULL = open(os.devnull, 'w')
    cmd = [
        "python", "-c",
        "import paddle.fluid as fluid; fleet_ptr = fluid.core.Fleet(); [fleet_ptr.copy_table_by_feasign(10, 10, [2020, 1010])];"
    ]
    proc = subprocess.Popen(cmd, stdout=FNULL, stderr=FNULL, cwd=os.getcwd())
    ret = proc.wait()
    if ret == -11:
        return "PSLIB"
    else:
        return "TRANSPILER"


def set_runtime_envs(cluster_envs, engine_yaml):
    if cluster_envs is None:
        cluster_envs = {}

    envs.set_runtime_environs(cluster_envs)

    need_print = {}
    for k, v in os.environ.items():
        if k.startswith("train.trainer."):
            need_print[k] = v

    print(envs.pretty_print_envs(need_print, ("Runtime Envs", "Value")))


def single_train_engine(args):
    _envs = envs.load_yaml(args.model)
    run_extras = get_all_inters_from_yaml(args.model, ["runner."])

    mode = envs.get_runtime_environ("mode")
    trainer_class = ".".join(["runner", mode, "trainer_class"])
    fleet_class = ".".join(["runner", mode, "fleet_mode"])
    device_class = ".".join(["runner", mode, "device"])
    selected_gpus_class = ".".join(["runner", mode, "selected_gpus"])

    trainer = run_extras.get(trainer_class, "GeneralTrainer")
    fleet_mode = run_extras.get(fleet_class, "ps")
    device = run_extras.get(device_class, "cpu")
    selected_gpus = run_extras.get(selected_gpus_class, "0")
    executor_mode = "train"

    single_envs = {}

    if device.upper() == "GPU":
        selected_gpus_num = len(selected_gpus.split(","))
        if selected_gpus_num != 1:
            raise ValueError(
                "Single Mode Only Support One GPU, Set Local Cluster Mode to use Multi-GPUS"
            )

        single_envs["selsected_gpus"] = selected_gpus
        single_envs["FLAGS_selected_gpus"] = selected_gpus

    single_envs["train.trainer.trainer"] = trainer
    single_envs["fleet_mode"] = fleet_mode
    single_envs["train.trainer.executor_mode"] = executor_mode
    single_envs["train.trainer.threads"] = "2"
    single_envs["train.trainer.platform"] = envs.get_platform()
    single_envs["train.trainer.engine"] = "single"

    set_runtime_envs(single_envs, args.model)
    trainer = TrainerFactory.create(args.model)
    return trainer


def single_infer_engine(args):
    _envs = envs.load_yaml(args.model)
    run_extras = get_all_inters_from_yaml(args.model, ["runner."])

    mode = envs.get_runtime_environ("mode")
    trainer_class = ".".join(["runner", mode, "trainer_class"])
    fleet_class = ".".join(["runner", mode, "fleet_mode"])
    device_class = ".".join(["runner", mode, "device"])
    selected_gpus_class = ".".join(["runner", mode, "selected_gpus"])

    trainer = run_extras.get(trainer_class, "GeneralTrainer")
    fleet_mode = run_extras.get(fleet_class, "ps")
    device = run_extras.get(device_class, "cpu")
    selected_gpus = run_extras.get(selected_gpus_class, "0")
    executor_mode = "infer"

    single_envs = {}

    if device.upper() == "GPU":
        selected_gpus_num = len(selected_gpus.split(","))
        if selected_gpus_num != 1:
            raise ValueError(
                "Single Mode Only Support One GPU, Set Local Cluster Mode to use Multi-GPUS"
            )

        single_envs["selsected_gpus"] = selected_gpus
        single_envs["FLAGS_selected_gpus"] = selected_gpus

    single_envs["train.trainer.trainer"] = trainer
    single_envs["train.trainer.executor_mode"] = executor_mode
    single_envs["fleet_mode"] = fleet_mode
    single_envs["train.trainer.threads"] = "2"
    single_envs["train.trainer.platform"] = envs.get_platform()
    single_envs["train.trainer.engine"] = "single"

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
        _envs = envs.load_yaml(args.backend)
        flattens = envs.flatten_environs(_envs, "_")
        flattens["engine_role"] = role
        flattens["engine_run_config"] = args.model
        flattens["engine_temp_path"] = tempfile.mkdtemp()
        update_workspace(flattens)

        envs.set_runtime_environs(flattens)
        print(envs.pretty_print_envs(flattens, ("Submit Envs", "Value")))

        launch = ClusterEngine(None, args.model)
        return launch

    def worker():
        role = "WORKER"

        _envs = envs.load_yaml(args.model)
        run_extras = get_all_inters_from_yaml(args.model,
                                              ["train.", "runner."])
        trainer_class = run_extras.get(
            "runner." + _envs["mode"] + ".trainer_class", None)

        if trainer_class:
            trainer = trainer_class
        else:
            trainer = "GeneralTrainer"

        executor_mode = "train"

        distributed_strategy = run_extras.get(
            "runner." + _envs["mode"] + ".distribute_strategy", "async")
        selected_gpus = run_extras.get(
            "runner." + _envs["mode"] + ".selected_gpus", "0")
        fleet_mode = run_extras.get("runner." + _envs["mode"] + ".fleet_mode",
                                    "ps")

        cluster_envs = {}
        cluster_envs["selected_gpus"] = selected_gpus
        cluster_envs["fleet_mode"] = fleet_mode
        cluster_envs["train.trainer.trainer"] = trainer
        cluster_envs["train.trainer.executor_mode"] = executor_mode
        cluster_envs["train.trainer.engine"] = "cluster"
        cluster_envs["train.trainer.strategy"] = distributed_strategy
        cluster_envs["train.trainer.threads"] = envs.get_runtime_environ(
            "CPU_NUM")
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
    print("launch cluster engine with cluster to run model: {}".format(
        args.model))

    cluster_envs = {}
    cluster_envs["train.trainer.trainer"] = "CtrCodingTrainer"
    cluster_envs["train.trainer.platform"] = envs.get_platform()

    set_runtime_envs(cluster_envs, args.model)

    trainer = TrainerFactory.create(args.model)
    return trainer


def local_cluster_engine(args):
    from paddlerec.core.engine.local_cluster import LocalClusterEngine

    _envs = envs.load_yaml(args.model)
    run_extras = get_all_inters_from_yaml(args.model, ["train.", "runner."])
    trainer_class = run_extras.get("runner." + _envs["mode"] + ".runner_class",
                                   None)

    if trainer_class:
        trainer = trainer_class
    else:
        trainer = "GeneralTrainer"

    executor_mode = "train"
    distributed_strategy = run_extras.get(
        "runner." + _envs["mode"] + ".distribute_strategy", "async")

    worker_num = run_extras.get("runner." + _envs["mode"] + ".worker_num", 1)
    server_num = run_extras.get("runner." + _envs["mode"] + ".server_num", 1)
    selected_gpus = run_extras.get(
        "runner." + _envs["mode"] + ".selected_gpus", "0")

    fleet_mode = run_extras.get("runner." + _envs["mode"] + ".fleet_mode", "")
    if fleet_mode == "":
        device = run_extras.get("runner." + _envs["mode"] + ".device", "cpu")
        if len(selected_gpus.split(",")) > 1 and device.upper() == "GPU":
            fleet_mode = "COLLECTIVE"
        else:
            fleet_mode = "PS"

    cluster_envs = {}
    cluster_envs["server_num"] = server_num
    cluster_envs["worker_num"] = worker_num
    cluster_envs["selected_gpus"] = selected_gpus
    cluster_envs["start_port"] = envs.find_free_port()
    cluster_envs["fleet_mode"] = fleet_mode
    cluster_envs["log_dir"] = "logs"
    cluster_envs["train.trainer.trainer"] = trainer
    cluster_envs["train.trainer.executor_mode"] = executor_mode
    cluster_envs["train.trainer.strategy"] = distributed_strategy
    cluster_envs["train.trainer.threads"] = "2"
    cluster_envs["train.trainer.engine"] = "local_cluster"
    cluster_envs["train.trainer.platform"] = envs.get_platform()

    cluster_envs["CPU_NUM"] = "2"
    print("launch {} engine with cluster to run model: {}".format(trainer,
                                                                  args.model))

    set_runtime_envs(cluster_envs, args.model)
    launch = LocalClusterEngine(cluster_envs, args.model)
    return launch


def local_mpi_engine(args):
    print("launch cluster engine with cluster to run model: {}".format(
        args.model))
    from paddlerec.core.engine.local_mpi import LocalMPIEngine

    print("use 1X1 MPI ClusterTraining at localhost to run model: {}".format(
        args.model))

    mpi = util.run_which("mpirun")
    if not mpi:
        raise RuntimeError("can not find mpirun, please check environment")

    _envs = envs.load_yaml(args.model)
    run_extras = get_all_inters_from_yaml(args.model, ["train.", "runner."])
    trainer_class = run_extras.get("runner." + _envs["mode"] + ".runner_class",
                                   None)
    executor_mode = "train"
    distributed_strategy = run_extras.get(
        "runner." + _envs["mode"] + ".distribute_strategy", "async")
    fleet_mode = run_extras.get("runner." + _envs["mode"] + ".fleet_mode",
                                "ps")

    if trainer_class:
        trainer = trainer_class
    else:
        trainer = "GeneralTrainer"

    cluster_envs = {}
    cluster_envs["mpirun"] = mpi
    cluster_envs["train.trainer.trainer"] = trainer
    cluster_envs["log_dir"] = "logs"
    cluster_envs["train.trainer.engine"] = "local_cluster"
    cluster_envs["train.trainer.executor_mode"] = executor_mode
    cluster_envs["fleet_mode"] = fleet_mode
    cluster_envs["train.trainer.strategy"] = distributed_strategy
    cluster_envs["train.trainer.threads"] = "2"
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

    if not validation.yaml_validation(args.model):
        sys.exit(-1)
    engine_registry()

    running_config = get_all_inters_from_yaml(args.model, ["mode", "runner."])
    modes = get_modes(running_config)

    for mode in modes:
        envs.set_runtime_environs({"mode": mode})
        which_engine = get_engine(args, running_config, mode)
        engine = which_engine(args)
        engine.run()
