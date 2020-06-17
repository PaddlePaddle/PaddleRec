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
import warnings

import copy
from paddlerec.core.factory import TrainerFactory
from paddlerec.core.utils import envs
from paddlerec.core.utils import util
from paddlerec.core.utils import validation

engines = {}
device = ["CPU", "GPU"]

engine_choices = ["TRAIN", "INFER", "LOCAL_CLUSTER_TRAIN", "CLUSTER_TRAIN"]


def engine_registry():
    engines["TRANSPILER"] = {}
    engines["PSLIB"] = {}

    engines["TRANSPILER"]["TRAIN"] = single_train_engine
    engines["TRANSPILER"]["INFER"] = single_infer_engine
    engines["TRANSPILER"]["LOCAL_CLUSTER_TRAIN"] = local_cluster_engine
    engines["TRANSPILER"]["CLUSTER"] = cluster_engine
    engines["PSLIB"]["TRAIN"] = local_mpi_engine
    engines["PSLIB"]["LOCAL_CLUSTER_TRAIN"] = local_mpi_engine
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

    engine_class = ".".join(["runner", mode, "class"])
    engine_device = ".".join(["runner", mode, "device"])
    device_gpu_choices = ".".join(["runner", mode, "selected_gpus"])

    engine = running_config.get(engine_class, None)
    if engine is None:
        raise ValueError("not find {} in yaml, please check".format(
            mode, engine_class))
    device = running_config.get(engine_device, None)

    engine = engine.upper()
    device = device.upper()

    if device is None:
        print("not find device be specified in yaml, set CPU as default")
        device = "CPU"

    if device == "GPU":
        selected_gpus = running_config.get(device_gpu_choices, None)

        if selected_gpus is None:
            print(
                "not find selected_gpus be specified in yaml, set `0` as default"
            )
            selected_gpus = "0"
        else:
            print("selected_gpus {} will be specified for running".format(
                selected_gpus))

        selected_gpus_num = len(selected_gpus.split(","))
        if selected_gpus_num > 1:
            engine = "LOCAL_CLUSTER_TRAIN"

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
    run_extras = get_all_inters_from_yaml(args.model, ["runner."])

    mode = envs.get_runtime_environ("mode")
    trainer_class = ".".join(["runner", mode, "trainer_class"])
    fleet_class = ".".join(["runner", mode, "fleet_mode"])
    device_class = ".".join(["runner", mode, "device"])
    selected_gpus_class = ".".join(["runner", mode, "selected_gpus"])

    epochs_class = ".".join(["runner", mode, "epochs"])
    epochs = run_extras.get(epochs_class, 1)
    if epochs > 1:
        warnings.warn(
            "It makes no sense to predict the same model for multiple epochs",
            category=UserWarning,
            stacklevel=2)

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
    def master():
        from paddlerec.core.engine.cluster.cluster import ClusterEngine
        _envs = envs.load_yaml(args.backend)
        flattens = envs.flatten_environs(_envs, "_")
        flattens["engine_role"] = "MASTER"
        flattens["engine_mode"] = envs.get_runtime_environ("mode")
        flattens["engine_run_config"] = args.model
        flattens["engine_temp_path"] = tempfile.mkdtemp()
        envs.set_runtime_environs(flattens)
        ClusterEngine.workspace_replace()
        print(envs.pretty_print_envs(flattens, ("Submit Envs", "Value")))

        launch = ClusterEngine(None, args.model)
        return launch

    def worker(mode):
        if not mode:
            raise ValueError("mode: {} can not be recognized")

        run_extras = get_all_inters_from_yaml(args.model, ["runner."])

        trainer_class = ".".join(["runner", mode, "trainer_class"])
        fleet_class = ".".join(["runner", mode, "fleet_mode"])
        device_class = ".".join(["runner", mode, "device"])
        selected_gpus_class = ".".join(["runner", mode, "selected_gpus"])
        strategy_class = ".".join(["runner", mode, "distribute_strategy"])
        worker_class = ".".join(["runner", mode, "worker_num"])
        server_class = ".".join(["runner", mode, "server_num"])

        trainer = run_extras.get(trainer_class, "GeneralTrainer")
        fleet_mode = run_extras.get(fleet_class, "ps")
        device = run_extras.get(device_class, "cpu")
        selected_gpus = run_extras.get(selected_gpus_class, "0")
        distributed_strategy = run_extras.get(strategy_class, "async")
        worker_num = run_extras.get(worker_class, 1)
        server_num = run_extras.get(server_class, 1)
        executor_mode = "train"

        device = device.upper()
        fleet_mode = fleet_mode.upper()

        if fleet_mode == "COLLECTIVE" and device != "GPU":
            raise ValueError("COLLECTIVE can not be used with GPU")

        cluster_envs = {}

        if device == "GPU":
            cluster_envs["selected_gpus"] = selected_gpus

        cluster_envs["server_num"] = server_num
        cluster_envs["worker_num"] = worker_num
        cluster_envs["fleet_mode"] = fleet_mode
        cluster_envs["train.trainer.trainer"] = trainer
        cluster_envs["train.trainer.engine"] = "cluster"
        cluster_envs["train.trainer.executor_mode"] = executor_mode
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
        mode = os.getenv("PADDLE_PADDLEREC_MODE", None)
        return worker(mode)
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
    def get_worker_num(run_extras, workers):
        _envs = envs.load_yaml(args.model)
        mode = envs.get_runtime_environ("mode")
        workspace = envs.get_runtime_environ("workspace")
        phases_class = ".".join(["runner", mode, "phases"])
        phase_names = run_extras.get(phases_class)
        phases = []
        all_phases = _envs.get("phase")
        if phase_names is None:
            phases = all_phases
        else:
            for phase in all_phases:
                if phase["name"] in phase_names:
                    phases.append(phase)

        dataset_names = []
        for phase in phases:
            dataset_names.append(phase["dataset_name"])

        datapaths = []
        for dataset in _envs.get("dataset"):
            if dataset["name"] in dataset_names:
                datapaths.append(dataset["data_path"])

        if not datapaths:
            raise ValueError("data path must exist for training/inference")

        datapaths = [
            envs.workspace_adapter_by_specific(path, workspace)
            for path in datapaths
        ]

        all_workers = [len(os.listdir(path)) for path in datapaths]
        all_workers.append(workers)
        max_worker_num = min(all_workers)

        if max_worker_num >= workers:
            return workers

        print(
            "phases do not have enough datas for training, set worker/gpu cards num from {} to {}".
            format(workers, max_worker_num))

        return max_worker_num

    from paddlerec.core.engine.local_cluster import LocalClusterEngine

    run_extras = get_all_inters_from_yaml(args.model, ["runner."])
    mode = envs.get_runtime_environ("mode")
    trainer_class = ".".join(["runner", mode, "trainer_class"])
    fleet_class = ".".join(["runner", mode, "fleet_mode"])
    device_class = ".".join(["runner", mode, "device"])
    selected_gpus_class = ".".join(["runner", mode, "selected_gpus"])
    strategy_class = ".".join(["runner", mode, "distribute_strategy"])
    worker_class = ".".join(["runner", mode, "worker_num"])
    server_class = ".".join(["runner", mode, "server_num"])

    trainer = run_extras.get(trainer_class, "GeneralTrainer")
    fleet_mode = run_extras.get(fleet_class, "ps")
    device = run_extras.get(device_class, "cpu")
    selected_gpus = run_extras.get(selected_gpus_class, "0")
    distributed_strategy = run_extras.get(strategy_class, "async")
    executor_mode = "train"

    worker_num = run_extras.get(worker_class, 1)
    server_num = run_extras.get(server_class, 1)

    device = device.upper()
    fleet_mode = fleet_mode.upper()

    cluster_envs = {}

    # Todo: delete follow hard code when paddle support ps-gpu.
    if device == "CPU":
        fleet_mode = "PS"
    elif device == "GPU":
        fleet_mode = "COLLECTIVE"
    if fleet_mode == "PS" and device != "CPU":
        raise ValueError("PS can not be used with GPU")

    if fleet_mode == "COLLECTIVE" and device != "GPU":
        raise ValueError("COLLECTIVE can not be used without GPU")

    if fleet_mode == "PS":
        worker_num = get_worker_num(run_extras, worker_num)

    if fleet_mode == "COLLECTIVE":
        cluster_envs["selected_gpus"] = selected_gpus
        gpus = selected_gpus.split(",")
        gpu_num = get_worker_num(run_extras, len(gpus))
        cluster_envs["selected_gpus"] = ','.join(gpus[:gpu_num])

    cluster_envs["server_num"] = server_num
    cluster_envs["worker_num"] = worker_num
    cluster_envs["start_port"] = envs.find_free_port()
    cluster_envs["fleet_mode"] = fleet_mode
    cluster_envs["log_dir"] = "logs"
    cluster_envs["train.trainer.trainer"] = trainer
    cluster_envs["train.trainer.executor_mode"] = executor_mode
    cluster_envs["train.trainer.strategy"] = distributed_strategy
    cluster_envs["train.trainer.threads"] = "2"
    cluster_envs["CPU_NUM"] = cluster_envs["train.trainer.threads"]
    cluster_envs["train.trainer.engine"] = "local_cluster"
    cluster_envs["train.trainer.platform"] = envs.get_platform()

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

    run_extras = get_all_inters_from_yaml(args.model, ["runner."])

    mode = envs.get_runtime_environ("mode")
    trainer_class = ".".join(["runner", mode, "trainer_class"])
    fleet_class = ".".join(["runner", mode, "fleet_mode"])
    distributed_strategy = "async"
    executor_mode = "train"

    trainer = run_extras.get(trainer_class, "GeneralTrainer")
    fleet_mode = run_extras.get(fleet_class, "ps")

    cluster_envs = {}
    cluster_envs["mpirun"] = mpi
    cluster_envs["train.trainer.trainer"] = trainer
    cluster_envs["log_dir"] = "logs"
    cluster_envs["train.trainer.engine"] = "local_cluster"
    cluster_envs["train.trainer.executor_mode"] = executor_mode
    cluster_envs["fleet_mode"] = fleet_mode
    cluster_envs["train.trainer.strategy"] = distributed_strategy
    cluster_envs["train.trainer.threads"] = "2"
    cluster_envs["train.trainer.platform"] = envs.get_platform()

    set_runtime_envs(cluster_envs, args.model)
    launch = LocalMPIEngine(cluster_envs, args.model)
    return launch


def get_abs_model(model):
    if model.startswith("paddlerec."):
        dir = envs.paddlerec_adapter(model)
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
    args.model = get_abs_model(args.model)

    if not validation.yaml_validation(args.model):
        sys.exit(-1)

    engine_registry()
    running_config = get_all_inters_from_yaml(
        args.model, ["workspace", "mode", "runner."])
    modes = get_modes(running_config)

    for mode in modes:
        envs.set_runtime_environs({
            "mode": mode,
            "workspace": running_config["workspace"]
        })
        which_engine = get_engine(args, running_config, mode)
        engine = which_engine(args)
        engine.run()
