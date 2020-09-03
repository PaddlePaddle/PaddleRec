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

import functools
import logging
import os
import copy
import socket
import sys
import subprocess
import socket

logger = logging.getLogger("root")
logger.propagate = False


def get_cluster(node_ips, node_ip, paddle_ports, selected_gpus):
    assert type(paddle_ports) is list, "paddle_ports must be list"
    cluster = Cluster(hdfs=None)
    trainer_rank = 0
    for node_rank, ip in enumerate(node_ips):
        pod = Pod()
        pod.rank = node_rank
        pod.addr = ip
        for i in range(len(selected_gpus)):
            trainer = Trainer()
            trainer.gpus.append(selected_gpus[i])
            trainer.endpoint = "%s:%d" % (ip, paddle_ports[i])
            trainer.rank = trainer_rank
            trainer_rank += 1

            pod.trainers.append(trainer)
        cluster.pods.append(pod)

    pod_rank = node_ips.index(node_ip)
    return cluster, cluster.pods[pod_rank]


def get_cloud_cluster(selected_gpus, args_port=None):
    #you can automatically get ip info while using paddlecloud multi nodes mode.
    node_ips = os.getenv("PADDLE_TRAINERS")
    assert node_ips is not None, "PADDLE_TRAINERS should not be None"
    print("node_ips:{}".format(node_ips))
    node_ip = os.getenv("POD_IP")
    assert node_ip is not None, "POD_IP should not be None"
    print("node_ip:{}".format(node_ip))
    node_rank = os.getenv("PADDLE_TRAINER_ID")
    assert node_rank is not None, "PADDLE_TRAINER_ID should not be None"
    print("node_rank:{}".format(node_rank))
    node_ips = node_ips.split(",")
    num_nodes = len(node_ips)
    node_rank = int(node_rank)

    started_port = args_port
    print("num_nodes:", num_nodes)
    if num_nodes > 1:
        try:
            paddle_port = int(os.getenv("PADDLE_PORT", ""))
            paddle_port_num = int(os.getenv("TRAINER_PORTS_NUM", ""))

            if paddle_port_num >= len(
                    selected_gpus) and paddle_port != args_port:
                logger.warning("Use Cloud specified port:{}.".format(
                    paddle_port))
                started_port = paddle_port

        except Exception as e:
            print(e)
            pass

    if started_port is None:
        started_port = 6170

    logger.debug("parsed from args:node_ips:{} \
        node_ip:{} node_rank:{} started_port:{}"
                 .format(node_ips, node_ip, node_rank, started_port))

    ports = [x for x in range(started_port, started_port + len(selected_gpus))]
    cluster, pod = get_cluster(node_ips, node_ip, ports, selected_gpus)
    return cluster, cluster.pods[node_rank]


def use_paddlecloud():
    node_ips = os.getenv("PADDLE_TRAINERS", None)
    node_ip = os.getenv("POD_IP", None)
    node_rank = os.getenv("PADDLE_TRAINER_ID", None)
    if node_ips is None or node_ip is None or node_rank is None:
        return False
    else:
        return True


class TrainerProc(object):
    def __init__(self):
        self.proc = None
        self.log_fn = None
        self.log_offset = None
        self.rank = None
        self.local_rank = None
        self.cmd = None


def start_local_trainers(cluster, pod, cmd, log_dir=None):
    current_env = copy.copy(os.environ.copy())
    #paddle broadcast ncclUniqueId use socket, and
    #proxy maybe make trainers unreachable, so delete them.
    #if we set them to "", grpc will log error message "bad uri"
    #so just delete them.
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    for idx, t in enumerate(pod.trainers):
        proc_env = {
            "FLAGS_selected_gpus": "%s" % ",".join([str(g) for g in t.gpus]),
            "PADDLE_TRAINER_ID": "%d" % t.rank,
            "PADDLE_CURRENT_ENDPOINT": "%s" % t.endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % cluster.trainers_nranks(),
            "PADDLE_TRAINER_ENDPOINTS": ",".join(cluster.trainers_endpoints())
        }

        current_env.update(proc_env)

        logger.debug("trainer proc env:{}".format(current_env))

        # cmd = [sys.executable, "-u", training_script]

        logger.info("start trainer proc:{} env:{}".format(cmd, proc_env))

        fn = None
        if log_dir is not None:
            os.system("mkdir -p {}".format(log_dir))
            fn = open("%s/workerlog.%d" % (log_dir, idx), "a")
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            proc = subprocess.Popen(cmd, env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = t.rank
        tp.local_rank = idx
        tp.log_fn = fn
        tp.log_offset = fn.tell() if fn else None
        tp.cmd = cmd

        procs.append(tp)

    return procs
