#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# !/bin/env python
import os
import sys
import time
import socket
from kubernetes import client, config

NAMESPACE = os.getenv("NAMESPACE")
if os.getenv("KUBERNETES_SERVICE_HOST", None):
    config.load_incluster_config()
else:
    config.load_kube_config()
v1 = client.CoreV1Api()


def get_pod_status(item):
    phase = item.status.phase

    # check terminate time although phase is Running.
    if item.metadata.deletion_timestamp != None:
        return "Terminating"

    return phase


def containers_all_ready(label_selector):
    def container_statuses_ready(item):
        container_statuses = item.status.container_statuses

        for status in container_statuses:
            if not status.ready:
                return False
        return True

    api_response = v1.list_namespaced_pod(
        namespace=NAMESPACE, pretty=True, label_selector=label_selector)

    for item in api_response.items:
        if not container_statuses_ready(item):
            return False

    return True


def fetch_pods_info(label_selector, phase=None):
    api_response = v1.list_namespaced_pod(
        namespace=NAMESPACE, pretty=True, label_selector=label_selector)
    pod_list = []
    for item in api_response.items:
        if phase is not None and get_pod_status(item) != phase:
            continue

        pod_list.append(
            (item.status.phase, item.status.pod_ip, item.metadata.name))
    return pod_list


def wait_pods_running(label_selector, desired):
    print("label selector: %s, desired: %s" % (label_selector, desired))
    while True:
        count = count_pods_by_phase(label_selector, 'Running')
        # NOTE: pods may be scaled.
        if count >= int(desired):
            break
        print('current cnt: %d sleep for 5 seconds...' % count)
        time.sleep(5)


def wait_containers_ready(label_selector):
    print("label selector: %s, wait all containers ready" % (label_selector))
    while True:
        if containers_all_ready(label_selector):
            break
        print('not all containers ready, sleep for 5 seconds...')
        time.sleep(5)


def count_pods_by_phase(label_selector, phase):
    pod_list = fetch_pods_info(label_selector, phase)
    return len(pod_list)


def fetch_ips_list(label_selector, phase=None):
    pod_list = fetch_pods_info(label_selector, phase)
    ips = [item[1] for item in pod_list]
    ips.sort()
    return ips


def fetch_name_list(label_selector, phase=None):
    pod_list = fetch_pods_info(label_selector, phase)
    names = [item[2] for item in pod_list]
    names.sort()
    return names


def fetch_ips_string(label_selector, phase=None):
    ips = fetch_ips_list(label_selector, phase)
    return ",".join(ips)


def fetch_endpoints_string(label_selector, port, phase=None, sameport=True):
    ips = fetch_ips_list(label_selector, phase)
    if sameport:
        ips = ["{0}:{1}".format(ip, port) for ip in ips]
    else:
        srcips = ips
        ips = []
        port = int(port)
        for ip in srcips:
            ips.append("{0}:{1}".format(ip, port))
            port = port + 1
    return ",".join(ips)


def fetch_pod_id(label_selector, phase=None, byname=True):
    if byname:
        names = fetch_name_list(label_selector, phase=phase)

        local_name = os.getenv('POD_NAME')
        for i in xrange(len(names)):
            if names[i] == local_name:
                return i

        return None
    else:
        ips = fetch_ips_list(label_selector, phase=phase)

        local_ip = socket.gethostbyname(socket.gethostname())
        for i in xrange(len(ips)):
            if ips[i] == local_ip:
                return i

        # in minikube there can be one node only
        local_ip = os.getenv("POD_IP")
        for i in xrange(len(ips)):
            if ips[i] == local_ip:
                return i

        return None


def fetch_ips(label_selector):
    return fetch_ips_string(label_selector, phase="Running")


def fetch_endpoints(label_selector, port):
    return fetch_endpoints_string(
        label_selector, port=port, phase="Running", sameport=True)


def fetch_id(label_selector):
    return fetch_pod_id(label_selector, phase="Running")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "fetch_ips":
        print(fetch_ips(sys.argv[2]))
    if command == "fetch_ips_string":
        print(fetch_ips_string(sys.argv[2], sys.argv[3]))
    elif command == "fetch_endpoints":
        print(fetch_endpoints(sys.argv[2], sys.argv[3]))
    elif command == "fetch_id":
        print(fetch_id(sys.argv[2]))
    elif command == "count_pods_by_phase":
        print(count_pods_by_phase(sys.argv[2], sys.argv[3]))
    elif command == "wait_pods_running":
        wait_pods_running(sys.argv[2], sys.argv[3])
    elif command == "wait_containers_ready":
        wait_containers_ready(sys.argv[2])
