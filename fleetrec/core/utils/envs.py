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

global_envs = {}


def set_runtime_envions(envs):
    assert isinstance(envs, dict)

    for k, v in envs.items():
        os.environ[k] = str(v)


def get_runtime_envion(key):
    return os.getenv(key, None)


def get_trainer():
    train_mode = get_runtime_envion("trainer.trainer")
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
    _env_name = env_name if namespace is None else ".".join([namespace, env_name])
    return global_envs.get(_env_name, default_value)


def get_global_envs():
    return global_envs


def pretty_print_envs(envs, header=None):
    spacing = 5
    max_k = 45
    max_v = 20

    for k, v in envs.items():
        max_k = max(max_k, len(k))
        max_v = max(max_v, len(str(v)))

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
        draws += h_format.format("fleetrec Global Envs", "Value")

    draws += line + "\n"

    for k, v in envs.items():
        draws += l_format.format(k, " " * spacing, str(v))

    draws += border

    _str = "\n{}\n".format(draws)
    return _str


def lazy_instance(package, class_name):
    models = get_global_env("train.model.models")
    model_package = __import__(package, globals(), locals(), package.split("."))
    instance = getattr(model_package, class_name)
    return instance
