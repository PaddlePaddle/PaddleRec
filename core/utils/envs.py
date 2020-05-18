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
import copy
import sys
import socket
from contextlib import closing

global_envs = {}


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


def set_global_envs(envs):
    assert isinstance(envs, dict)

    def fatten_env_namespace(namespace_nests, local_envs):
        for k, v in local_envs.items():
            if isinstance(v, dict):
                nests = copy.deepcopy(namespace_nests)
                nests.append(k)
                fatten_env_namespace(nests, v)
            else:
                global_k = ".".join(namespace_nests + [k])
                global_envs[global_k] = v

    for k, v in envs.items():
        fatten_env_namespace([k], v)


def get_global_env(env_name, default_value=None, namespace=None):
    """
    get os environment value
    """
    _env_name = env_name if namespace is None else ".".join(
        [namespace, env_name])
    return global_envs.get(_env_name, default_value)


def get_global_envs():
    return global_envs


def windows_path_adapter(path):
    def adapt(l_p):
        if get_platform() == "WINDOWS":
            adapted_p = l_p.split("paddlerec.")[1].replace(".", "\\")
        else:
            adapted_p = l_p.split("paddlerec.")[1].replace(".", "/")
        return adapted_p

    if path.startswith("paddlerec."):
        package = get_runtime_environ("PACKAGE_BASE")
        return os.path.join(package, adapt(path))
    else:
        return adapt(path)


def update_workspace():
    workspace = global_envs.get("train.workspace", None)
    if not workspace:
        return
    path = windows_path_adapter(workspace)

    for name, value in global_envs.items():
        if isinstance(value, str):
            value = value.replace("{workspace}", path)
            global_envs[name] = value


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
    models = get_global_env("train.model.models")
    model_package = __import__(
        package, globals(), locals(), package.split("."))
    instance = getattr(model_package, class_name)
    return instance


def lazy_instance_by_fliename(abs, class_name):
    dirname = os.path.dirname(abs)
    sys.path.append(dirname)
    package = os.path.splitext(os.path.basename(abs))[0]

    model_package = __import__(
        package, globals(), locals(), package.split("."))
    instance = getattr(model_package, class_name)
    return instance


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
        with closing(socket.socket(socket.AF_INET,
                                   socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    new_port = __free_port()
    return new_port
