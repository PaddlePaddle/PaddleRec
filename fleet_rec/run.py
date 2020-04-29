import argparse
import os
import subprocess
import yaml

from fleetrec.core.factory import TrainerFactory
from fleetrec.core.utils import envs
from fleetrec.core.utils import util

engines = {}
device = ["CPU", "GPU"]
clusters = ["SINGLE", "LOCAL_CLUSTER", "CLUSTER"]


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


def get_engine(engine, device):
    d_engine = engines[device]
    transpiler = get_transpiler()
    run_engine = d_engine[transpiler].get(engine, None)

    if run_engine is None:
        raise ValueError("engine {} can not be supported on device: {}".format(engine, device))
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

    envs.set_runtime_environs(cluster_envs)
    envs.set_runtime_environs(get_engine_extras())

    need_print = {}
    for k, v in os.environ.items():
        if k.startswith("train.trainer."):
            need_print[k] = v

    print(envs.pretty_print_envs(need_print, ("Runtime Envs", "Value")))


def single_engine(args):
    print("use single engine to run model: {}".format(args.model))

    single_envs = {}
    single_envs["train.trainer.trainer"] = "SingleTrainer"
    single_envs["train.trainer.threads"] = "2"
    single_envs["train.trainer.engine"] = "single"
    single_envs["train.trainer.device"] = args.device
    single_envs["train.trainer.platform"] = envs.get_platform()

    set_runtime_envs(single_envs, args.model)
    trainer = TrainerFactory.create(args.model)
    return trainer


def cluster_engine(args):
    print("launch cluster engine with cluster to run model: {}".format(args.model))

    cluster_envs = {}
    cluster_envs["train.trainer.trainer"] = "ClusterTrainer"
    cluster_envs["train.trainer.engine"] = "cluster"
    cluster_envs["train.trainer.device"] = args.device
    cluster_envs["train.trainer.platform"] = envs.get_platform()

    set_runtime_envs(cluster_envs, args.model)

    trainer = TrainerFactory.create(args.model)
    return trainer


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
    print("launch cluster engine with cluster to run model: {}".format(args.model))
    from fleetrec.core.engine.local_cluster_engine import LocalClusterEngine

    cluster_envs = {}
    cluster_envs["server_num"] = 1
    cluster_envs["worker_num"] = 1
    cluster_envs["start_port"] = 36001
    cluster_envs["log_dir"] = "logs"
    cluster_envs["train.trainer.trainer"] = "ClusterTrainer"
    cluster_envs["train.trainer.strategy"] = "async"
    cluster_envs["train.trainer.threads"] = "2"
    cluster_envs["train.trainer.engine"] = "local_cluster"

    cluster_envs["train.trainer.device"] = args.device
    cluster_envs["train.trainer.platform"] = envs.get_platform()

    cluster_envs["CPU_NUM"] = "2"

    set_runtime_envs(cluster_envs, args.model)

    launch = LocalClusterEngine(cluster_envs, args.model)
    return launch


def local_mpi_engine(args):
    print("launch cluster engine with cluster to run model: {}".format(args.model))
    from fleetrec.core.engine.local_mpi_engine import LocalMPIEngine

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fleet-rec run')
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-e", "--engine", type=str, choices=["single", "local_cluster", "cluster"])
    parser.add_argument("-d", "--device", type=str, choices=["cpu", "gpu"], default="cpu")

    args = parser.parse_args()
    args.engine = args.engine.upper()
    args.device = args.device.upper()

    if not os.path.isfile(args.model):
        raise IOError("argument model: {} do not exist".format(args.model))
    engine_registry()

    which_engine = get_engine(args.engine, args.device)

    engine = which_engine(args)
    engine.run()
