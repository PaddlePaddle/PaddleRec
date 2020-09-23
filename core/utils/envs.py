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

from contextlib import closing
import yaml
import copy
import os
import socket
import sys
import six
import traceback
import warnings

global_envs = {}
global_envs_flatten = {}


def flatten_environs(envs, separator="."):
    flatten_dict = {}
    assert isinstance(envs, dict)

    def fatten_env_namespace(namespace_nests, local_envs):
        if not isinstance(local_envs, dict):
            global_k = separator.join(namespace_nests)
            flatten_dict[global_k] = str(local_envs)
        else:
            for k, v in local_envs.items():
                if isinstance(v, dict):
                    nests = copy.deepcopy(namespace_nests)
                    nests.append(k)
                    fatten_env_namespace(nests, v)
                else:
                    global_k = separator.join(namespace_nests + [k])
                    flatten_dict[global_k] = str(v)

    for k, v in envs.items():
        fatten_env_namespace([k], v)

    return flatten_dict


def set_runtime_environs(environs):
    for k, v in environs.items():
        os.environ[k] = str(v)


def get_runtime_environ(key):
    return os.getenv(key, None)


def get_trainer():
    train_mode = get_runtime_environ("train.trainer.trainer")
    return train_mode


def get_fleet_mode():
    fleet_mode = get_runtime_environ("fleet_mode")
    return fleet_mode


def set_global_envs(envs):
    assert isinstance(envs, dict)

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
                        raise ValueError("name must be in dataset list ", v)
                    nests = copy.deepcopy(namespace_nests)
                    nests.append(k)
                    nests.append(i["name"])
                    fatten_env_namespace(nests, i)
            else:
                global_k = ".".join(namespace_nests + [k])
                global_envs[global_k] = v

    fatten_env_namespace([], envs)

    for name, value in global_envs.items():
        if isinstance(value, str):
            value = os_path_adapter(workspace_adapter(value))
            global_envs[name] = value

    for runner in envs["runner"]:
        if "save_step_interval" in runner or "save_step_path" in runner:
            phase_name = runner["phases"]
            phase = [
                phase for phase in envs["phase"]
                if phase["name"] == phase_name[0]
            ]
            dataset_name = phase[0].get("dataset_name")
            dataset = [
                dataset for dataset in envs["dataset"]
                if dataset["name"] == dataset_name
            ]
            if dataset[0].get("type") == "QueueDataset":
                runner["save_step_interval"] = None
                runner["save_step_path"] = None
                warnings.warn(
                    "QueueDataset can not support save by step, please not config save_step_interval and save_step_path in your yaml"
                )

    if get_platform() != "LINUX":
        for dataset in envs["dataset"]:
            name = ".".join(["dataset", dataset["name"], "type"])
            global_envs[name] = "DataLoader"

    if get_platform() == "LINUX" and six.PY3:
        print("QueueDataset can not support PY3, change to DataLoader")
        for dataset in envs["dataset"]:
            name = ".".join(["dataset", dataset["name"], "type"])
            global_envs[name] = "DataLoader"


def get_global_env(env_name, default_value=None, namespace=None):
    """
    get os environment value
    """
    _env_name = env_name if namespace is None else ".".join(
        [namespace, env_name])
    return global_envs.get(_env_name, default_value)


def get_global_envs():
    return global_envs


def paddlerec_adapter(path):
    if path.startswith("paddlerec."):
        package = get_runtime_environ("PACKAGE_BASE")
        l_p = path.split("paddlerec.")[1].replace(".", "/")
        return os.path.join(package, l_p)
    else:
        return path


def os_path_adapter(value):
    if get_platform() == "WINDOWS":
        value = value.replace("/", "\\")
    else:
        value = value.replace("\\", "/")
    return value


def workspace_adapter(value):
    workspace = global_envs.get("workspace")
    return workspace_adapter_by_specific(value, workspace)


def workspace_adapter_by_specific(value, workspace):
    workspace = paddlerec_adapter(workspace)
    value = value.replace("{workspace}", workspace)
    return value


def reader_adapter():
    if get_platform() != "WINDOWS":
        return

    datasets = global_envs.get("dataset")
    for dataset in datasets:
        dataset["type"] = "DataLoader"


def pretty_print_envs(envs, header=None):
    spacing = 5
    max_k = 45
    max_v = 50

    for k, v in envs.items():
        max_k = max(max_k, len(k))

    h_format = "{{:^{}s}}{}{{:<{}s}}\n".format(max_k, " " * spacing, max_v)
    l_format = "{{:<{}s}}{{}}{{:<{}s}}\n".format(max_k, max_v)
    length = max_k + max_v + spacing

    border = "".join(["="] * length)
    line = "".join(["-"] * length)

    draws = ""
    draws += border + "\n"

    if header:
        draws += h_format.format(header[0], header[1])
    else:
        draws += h_format.format("paddlerec Global Envs", "Value")

    draws += line + "\n"

    for k, v in envs.items():
        if isinstance(v, str) and len(v) >= max_v:
            str_v = "... " + v[-46:]
        else:
            str_v = v

        draws += l_format.format(k, " " * spacing, str(str_v))

    draws += border

    _str = "\n{}\n".format(draws)
    return _str


def lazy_instance_by_package(package, class_name):
    try:
        model_package = __import__(package,
                                   globals(), locals(), package.split("."))
        instance = getattr(model_package, class_name)
        return instance
    except Exception as err:
        traceback.print_exc()
        print('Catch Exception:%s' % str(err))
        return None


def lazy_instance_by_fliename(abs, class_name):
    try:
        dirname = os.path.dirname(abs)
        sys.path.append(dirname)
        package = os.path.splitext(os.path.basename(abs))[0]

        model_package = __import__(package,
                                   globals(), locals(), package.split("."))
        instance = getattr(model_package, class_name)
        return instance
    except Exception as err:
        traceback.print_exc()
        print('Catch Exception:%s' % str(err))
        return None


def get_platform():
    import platform
    plats = platform.platform()
    if 'Linux' in plats:
        return "LINUX"
    if 'Darwin' in plats:
        return "DARWIN"
    if 'Windows' in plats:
        return "WINDOWS"


def find_free_port():
    def __free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    new_port = __free_port()
    return new_port


def load_yaml(config):
    vs = [int(i) for i in yaml.__version__.split(".")]
    if vs[0] < 5:
        use_full_loader = False
    elif vs[0] > 5:
        use_full_loader = True
    else:
        if vs[1] >= 1:
            use_full_loader = True
        else:
            use_full_loader = False

    if os.path.isfile(config):
        if six.PY2:
            with open(config, 'r') as rb:
                if use_full_loader:
                    _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
                else:
                    _config = yaml.load(rb.read())
                return _config
        else:
            with open(config, 'r', encoding="utf-8") as rb:
                if use_full_loader:
                    _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
                else:
                    _config = yaml.load(rb.read())
                return _config
    else:
        raise ValueError("config {} can not be supported".format(config))
