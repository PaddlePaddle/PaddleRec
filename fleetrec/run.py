import argparse
import os
import sys

import yaml
from paddle.fluid.incubate.fleet.parameter_server import version

from fleetrec.core.factory import TrainerFactory
from fleetrec.core.utils import envs
from fleetrec.core.engine import local_engine


def run(model_yaml):
    trainer = TrainerFactory.create(model_yaml)
    trainer.run()


def single_engine(single_envs, model_yaml):
    print(envs.pretty_print_envs(single_envs, ("Single Envs", "Value")))

    envs.set_runtime_envions(single_envs)
    run(model_yaml)


def local_cluster_engine(cluster_envs, model_yaml):
    print(envs.pretty_print_envs(cluster_envs, ("Local Cluster Envs", "Value")))
    envs.set_runtime_envions(cluster_envs)
    launch = local_engine.Launch(cluster_envs, model_yaml)
    launch.run()


def local_mpi_engine(model_yaml):
    print("use 1X1 MPI ClusterTraining at localhost to run model: {}".format(args.model))

    cluster_envs = {}
    cluster_envs["server_num"] = 1
    cluster_envs["worker_num"] = 1
    cluster_envs["start_port"] = 36001
    cluster_envs["log_dir"] = "logs"
    cluster_envs["train.trainer"] = "CtrTraining"

    print(envs.pretty_print_envs(cluster_envs, ("Local MPI Cluster Envs", "Value")))
    envs.set_runtime_envions(cluster_envs)
    print("coming soon")


def yaml_engine(engine_yaml, model_yaml):
    with open(engine_yaml, 'r') as rb:
        _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
    assert _config is not None

    envs.set_global_envs(_config)

    train_location = envs.get_global_env("engine.file")
    train_dirname = os.path.dirname(train_location)
    base_name = os.path.splitext(os.path.basename(train_location))[0]
    sys.path.append(train_dirname)
    trainer_class = envs.lazy_instance(base_name, "UserDefineTrainer")
    trainer = trainer_class(model_yaml)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fleet-rec run')
    parser.add_argument("--model", type=str)
    parser.add_argument("--engine", type=str)
    parser.add_argument("--engine_extras", type=str)

    args = parser.parse_args()

    if not os.path.exists(args.model) or not os.path.isfile(args.model):
        raise ValueError("argument model: {} error, must specify a existed yaml file".format(args.model))

    if args.engine.upper() == "SINGLE":
        if version.is_transpiler():
            print("use SingleTraining to run model: {}".format(args.model))
            single_envs = {"train.trainer": "SingleTraining"}
            single_engine(single_envs, args.model)
        else:
            local_mpi_engine(args.model)
    elif args.engine.upper() == "LOCAL_CLUSTER":
        print("use 1X1 ClusterTraining at localhost to run model: {}".format(args.model))
        if version.is_transpiler():
            cluster_envs = {}
            cluster_envs["server_num"] = 1
            cluster_envs["worker_num"] = 1
            cluster_envs["start_port"] = 36001
            cluster_envs["log_dir"] = "logs"
            cluster_envs["train.trainer"] = "ClusterTraining"
            cluster_envs["train.strategy.mode"] = "async"

            local_cluster_engine(cluster_envs, args.model)
        else:
            local_mpi_engine(args.model)
    elif args.engine.upper() == "CLUSTER":
        print("launch ClusterTraining with cluster to run model: {}".format(args.model))
        run(args.model)
    elif args.engine.upper() == "USER_DEFINE":
        engine_file = args.engine_extras
        if not os.path.exists(engine_file) or not os.path.isfile(engine_file):
            raise ValueError(
                "argument engine: user_define error, must specify a existed yaml file".format(args.engine_file))
        yaml_engine(engine_file, args.model)
    else:
        raise ValueError("engine only support SINGLE/LOCAL_CLUSTER/CLUSTER/USER_DEFINE")
