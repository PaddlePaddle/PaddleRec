import argparse
import os

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

    launch = local_engine.Launch(cluster_envs, model_yaml)
    launch.run()


def local_mpi_engine(cluster_envs, model_yaml):
    print(envs.pretty_print_envs(cluster_envs, ("Local MPI Cluster Envs", "Value")))
    print("coming soon")


def yaml_engine(engine_yaml, model_yaml):
    print("coming soon")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fleet-rec run')
    parser.add_argument("--model", type=str)
    parser.add_argument("--engine", type=str)

    args = parser.parse_args()

    if not os.path.exists(args.model) or not os.path.isfile(args.model):
        raise ValueError("argument model: {} error, must specify a existed yaml file".format(args.model))

    if args.engine == "Single":
        print("use SingleTraining to run model: {}".format(args.model))
        single_envs = {}
        single_envs["train.trainer"] = "SingleTraining"

        single_engine(single_envs, args.model)
    elif args.engine == "LocalCluster":
        print("use 1X1 ClusterTraining at localhost to run model: {}".format(args.model))

        cluster_envs = {}
        cluster_envs["server_num"] = 1
        cluster_envs["worker_num"] = 1
        cluster_envs["start_port"] = 36001
        cluster_envs["log_dir"] = "logs"
        cluster_envs["train.trainer"] = "ClusterTraining"

        local_cluster_engine(cluster_envs, args.model)
    elif args.engine == "LocalMPI":
        print("use 1X1 MPI ClusterTraining at localhost to run model: {}".format(args.model))

        cluster_envs = {}
        cluster_envs["server_num"] = 1
        cluster_envs["worker_num"] = 1
        cluster_envs["start_port"] = 36001
        cluster_envs["log_dir"] = "logs"
        cluster_envs["train.trainer"] = "CtrTraining"

        local_mpi_engine(cluster_envs, args.model)
    else:
        if not os.path.exists(args.engine) or not os.path.isfile(args.engine):
            raise ValueError("argument engine: {} error, must specify a existed yaml file".format(args.model))
        yaml_engine(args.engine, args.model)
