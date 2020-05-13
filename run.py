import argparse
import os
import subprocess
import tempfile

import yaml

from paddlerec.core.factory import TrainerFactory
from paddlerec.core.utils import envs
from paddlerec.core.utils import util

engines = {}
device = ["CPU", "GPU"]
clusters = ["SINGLE", "LOCAL_CLUSTER", "CLUSTER"]
custom_model = ['tdm']
model_name = ""


def engine_registry():
    cpu = {"TRANSPILER": {}, "PSLIB": {}}
    cpu["TRANSPILER"]["SINGLE"] = single_engine
    cpu["TRANSPILER"]["LOCAL_CLUSTER"] = local_cluster_engine
    cpu["TRANSPILER"]["CLUSTER"] = cluster_engine
    cpu["PSLIB"]["SINGLE"] = local_mpi_engine
    cpu["PSLIB"]["LOCAL_CLUSTER"] = local_mpi_engine
    cpu["PSLIB"]["CLUSTER"] = cluster_mpi_engine

    gpu = {"TRANSPILER": {}, "PSLIB": {}}
    gpu["TRANSPILER"]["SINGLE"] = single_engine

    engines["CPU"] = cpu
    engines["GPU"] = gpu


def get_engine(args):
    device = args.device
    d_engine = engines[device]
    transpiler = get_transpiler()

    engine = args.engine
    run_engine = d_engine[transpiler].get(engine, None)

    if run_engine is None:
        raise ValueError(
            "engine {} can not be supported on device: {}".format(engine, device))
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
    def get_engine_extras():
        with open(engine_yaml, 'r') as rb:
            _envs = yaml.load(rb.read(), Loader=yaml.FullLoader)

        flattens = envs.flatten_environs(_envs)

        engine_extras = {}
        for k, v in flattens.items():
            if k.startswith("train.trainer."):
                engine_extras[k] = v
        return engine_extras

    if cluster_envs is None:
        cluster_envs = {}

    engine_extras = get_engine_extras()
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
    single_envs["train.trainer.device"] = args.device
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

        # is fleet inner models
        if workspace.startswith("paddlerec."):
            fleet_package = envs.get_runtime_environ("PACKAGE_BASE")
            workspace_dir = workspace.split("paddlerec.")[1].replace(".", "/")
            path = os.path.join(fleet_package, workspace_dir)
        else:
            path = workspace

        for name, value in cluster_envs.items():
            if isinstance(value, str):
                value = value.replace("{workspace}", path)
                cluster_envs[name] = value

    def master():
        from paddlerec.core.engine.cluster.cluster import ClusterEngine
        with open(args.backend, 'r') as rb:
            _envs = yaml.load(rb.read(), Loader=yaml.FullLoader)

        flattens = envs.flatten_environs(_envs, "_")
        flattens["engine_role"] = args.role
        flattens["engine_temp_path"] = tempfile.mkdtemp()
        update_workspace(flattens)

        envs.set_runtime_environs(flattens)
        print(envs.pretty_print_envs(flattens, ("Submit Runtime Envs", "Value")))

        launch = ClusterEngine(None, args.model)
        return launch

    def worker():
        trainer = get_trainer_prefix(args) + "ClusterTrainer"
        cluster_envs = {}
        cluster_envs["train.trainer.trainer"] = trainer
        cluster_envs["train.trainer.engine"] = "cluster"
        cluster_envs["train.trainer.device"] = args.device
        cluster_envs["train.trainer.platform"] = envs.get_platform()
        print("launch {} engine with cluster to with model: {}".format(trainer, args.model))
        set_runtime_envs(cluster_envs, args.model)

        trainer = TrainerFactory.create(args.model)
        return trainer

    if args.role == "WORKER":
        return worker()
    else:
        return master()


def cluster_mpi_engine(args):
    print("launch cluster engine with cluster to run model: {}".format(args.model))

    cluster_envs = {}
    cluster_envs["train.trainer.trainer"] = "CtrCodingTrainer"
    cluster_envs["train.trainer.device"] = args.device
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

    cluster_envs["train.trainer.device"] = args.device
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

    cluster_envs["train.trainer.device"] = args.device
    cluster_envs["train.trainer.platform"] = envs.get_platform()

    set_runtime_envs(cluster_envs, args.model)
    launch = LocalMPIEngine(cluster_envs, args.model)
    return launch


def get_abs_model(model):
    if model.startswith("paddlerec."):
        fleet_base = envs.get_runtime_environ("PACKAGE_BASE")
        workspace_dir = model.split("paddlerec.")[1].replace(".", "/")
        path = os.path.join(fleet_base, workspace_dir, "config.yaml")
    else:
        if not os.path.isfile(model):
            raise IOError("model config: {} invalid".format(model))
        path = model
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='paddle-rec run')
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-e", "--engine", type=str,
                        choices=["single", "local_cluster", "cluster",
                                 "tdm_single", "tdm_local_cluster", "tdm_cluster"])

    parser.add_argument("-d", "--device", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("-b", "--backend", type=str, default=None)
    parser.add_argument("-r", "--role", type=str, choices=["master", "worker"], default="master")

    abs_dir = os.path.dirname(os.path.abspath(__file__))
    envs.set_runtime_environs({"PACKAGE_BASE": abs_dir})

    args = parser.parse_args()
    args.engine = args.engine.upper()
    args.device = args.device.upper()
    args.role = args.role.upper()

    model_name = args.model.split('.')[-1]
    args.model = get_abs_model(args.model)
    engine_registry()

    which_engine = get_engine(args)

    engine = which_engine(args)
    engine.run()
