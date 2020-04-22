import argparse
import os
import subprocess
import yaml

from fleetrec.core.factory import TrainerFactory
from fleetrec.core.utils import envs
from fleetrec.core.utils import util

engines = {"TRAINSPILER": {}, "PSLIB": {}}
clusters = ["SINGLE", "LOCAL_CLUSTER", "CLUSTER"]


def is_transpiler():
    FNULL = open(os.devnull, 'w')
    cmd = ["python", "-c",
           "import paddle.fluid as fluid; fleet_ptr = fluid.core.Fleet(); [fleet_ptr.copy_table_by_feasign(10, 10, [2020, 1010])];"]
    proc = subprocess.Popen(cmd, stdout=FNULL, stderr=FNULL, cwd=os.getcwd())
    ret = proc.wait()
    if ret == -11:
        return False
    else:
        return True


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


def get_engine(engine):
    engine = engine.upper()
    if version.is_transpiler():
        run_engine = engines["TRAINSPILER"].get(engine, None)
    else:
        run_engine = engines["PSLIB"].get(engine, None)

    if run_engine is None:
        raise ValueError("engine only support SINGLE/LOCAL_CLUSTER/CLUSTER")
    return run_engine


def single_engine(args):
    print("use single engine to run model: {}".format(args.model))

    single_envs = {}
    single_envs["train.trainer.trainer"] = "SingleTrainer"
    single_envs["train.trainer.threads"] = "2"
    single_envs["train.trainer.engine"] = "single"
    set_runtime_envs(single_envs, args.model)
    trainer = TrainerFactory.create(args.model)
    return trainer


def cluster_engine(args):
    print("launch cluster engine with cluster to run model: {}".format(args.model))

    cluster_envs = {}
    cluster_envs["train.trainer.trainer"] = "ClusterTrainer"
    cluster_envs["train.trainer.engine"] = "cluster"
    set_runtime_envs(cluster_envs, args.model)

    trainer = TrainerFactory.create(args.model)
    return trainer


def cluster_mpi_engine(args):
    print("launch cluster engine with cluster to run model: {}".format(args.model))

    cluster_envs = {}
    cluster_envs["train.trainer.trainer"] = "CtrCodingTrainer"
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

    set_runtime_envs(cluster_envs, args.model)
    launch = LocalMPIEngine(cluster_envs, args.model)
    return launch


def engine_registry():
    engines["TRAINSPILER"]["SINGLE"] = single_engine
    engines["TRAINSPILER"]["LOCAL_CLUSTER"] = local_cluster_engine
    engines["TRAINSPILER"]["CLUSTER"] = cluster_engine
    engines["PSLIB"]["SINGLE"] = local_mpi_engine
    engines["PSLIB"]["LOCAL_CLUSTER"] = local_mpi_engine
    engines["PSLIB"]["CLUSTER"] = cluster_mpi_engine


engine_registry()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fleet-rec run')
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-e", "--engine", type=str)

    args = parser.parse_args()

    if not os.path.exists(args.model) or not os.path.isfile(args.model):
        raise ValueError("argument model: {} error, must specify an existed YAML file".format(args.model))

    if args.engine.upper() not in clusters:
        raise ValueError("argument engine: {} error, must in {}".format(args.engine, clusters))

    which_engine = get_engine(args.engine)
    engine = which_engine(args)
    engine.run()
