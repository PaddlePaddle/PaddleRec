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


def encode_value(v):
    return v


def decode_value(v):
    return v


def set_global_envs(yaml):
    for k, v in yaml.items():
        os.environ[k] = encode_value(v)


def get_global_env(env_name, default_value=None):
    """
    get os environment value
    """
    if env_name not in os.environ:
        return default_value

    v = os.environ[env_name]
    return decode_value(v)
