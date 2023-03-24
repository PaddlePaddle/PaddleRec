# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ['DEBUG', 'DRY_RUN']


def get_flag_from_env_var(name, default):
    env_var = os.environ.get(name, None)
    if env_var in ('True', 'true', 'TRUE', '1'):
        return True
    elif env_var in ('False', 'false', 'FALSE', '0'):
        return False
    else:
        return default


DEBUG = get_flag_from_env_var('PADDLE_UAPI_DEBUG', False)
DRY_RUN = get_flag_from_env_var('PADDLE_UAPI_DRY_RUN', False)
CHECK_OPTS = get_flag_from_env_var('PADDLE_UAPI_CHECK_OPTS', False)
