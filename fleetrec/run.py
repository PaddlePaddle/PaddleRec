import argparse
import os

import yaml
from paddle.fluid.incubate.fleet.parameter_server import version

from fleetrec.core.factory import TrainerFactory
from fleetrec.core.utils import envs
from fleetrec.core.utils import util

engines = {"TRAINSPILER": {}, "PSLIB": {}}
clusters = ["SINGLE", "LOCAL_CLUSTER", "CLUSTER"]


def set_runtime_envs(cluster_envs, engine_yaml):
    if engine_yaml is not None:
        with open(engine_yaml, 'r') as rb:
            _envs = yaml.load(rb.read(), Loader=yaml.FullLoader)
    else:
        _envs = {}

    if cluster_envs is None:
        cluster_envs = {}
    cluster_envs.update(_envs)
    envs.set_runtime_envions(cluster_envs)
    print(envs.pretty_print_envs(cluster_envs, ("Runtime Envs", "Value")))


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
    single_envs = {"trainer.trainer": "SingleTrainer", "trainer.threads": "2"}
    set_runtime_envs(single_envs, args.engine_extras)
    trainer = TrainerFactory.create(args.model)
    return trainer


def cluster_engine(args):
    print("launch cluster engine with cluster to run model: {}".format(args.model))

    cluster_envs = {"trainer.trainer": "ClusterTrainer"}
    set_runtime_envs(cluster_envs, args.engine_extras)

    envs.set_runtime_envions(cluster_envs)
    trainer = TrainerFactory.create(args.model)
    return trainer


def cluster_mpi_engine(args):
    print("launch cluster engine with cluster to run model: {}".format(args.model))

    cluster_envs = {"trainer.trainer": "CtrCodingTrainer"}
    set_runtime_envs(cluster_envs, args.engine_extras)

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
    cluster_envs["trainer.trainer"] = "ClusterTrainer"
    cluster_envs["trainer.strategy.mode"] = "async"

    set_runtime_envs(cluster_envs, args.engine_extras)

    launch = LocalClusterEngine(cluster_envs, args.model)
    return launch


def local_mpi_engine(args):
    print("launch cluster engine with cluster to run model: {}".format(args.model))
    from fleetrec.core.engine.local_mpi_engine import LocalMPIEngine

    print("use 1X1 MPI ClusterTraining at localhost to run model: {}".format(args.model))

    mpi = util.run_which("mpirun")
    if not mpi:
        raise RuntimeError("can not find mpirun, please check environment")

    cluster_envs = {"mpirun": mpi, "trainer.trainer": "CtrCodingTrainer", "log_dir": "logs"}
    set_runtime_envs(cluster_envs, args.engine_extras)
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
    parser.add_argument("-ex", "--engine_extras", default=None, type=str)

    args = parser.parse_args()

    if not os.path.exists(args.model) or not os.path.isfile(args.model):
        raise ValueError("argument model: {} error, must specify an existed YAML file".format(args.model))

    if args.engine.upper() not in clusters:
        raise ValueError("argument engine: {} error, must in {}".format(args.engine, clusters))

    if args.engine_extras is not None:
        if not os.path.exists(args.engine_extras) or not os.path.isfile(args.engine_extras):
            raise ValueError(
                "argument engine_extras: {} error, must specify an existed YAML file".format(args.engine_extras))

    which_engine = get_engine(args.engine)
    engine = which_engine(args)
    engine.run()
