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

from collections import OrderedDict

# TODO: We need a lightweight RDBMS to handle these tables.


class Registry(object):
    def __init__(self, required_keys, primary_key):
        super().__init__()
        self._table = OrderedDict()
        self.required_keys = required_keys
        self.primary_key = primary_key
        assert self.primary_key in self.required_keys

    def register_record(self, record, validate=True):
        if validate:
            self._validate_record(record)
        prim = record[self.primary_key]
        self._table[prim] = record

    def _validate_record(self, record):
        for key in self.required_keys:
            if key not in record:
                raise KeyError(f"Key {repr(key)} is required, but not found.")

    def query(self, prim_key):
        return self._table[prim_key]

    def all_records(self):
        yield from self._table.items()

    def __str__(self):
        # TODO: Tabulate records in prettier format
        return str(self._table)


def build_runner_from_model_info(model_info):
    suite_name = model_info['suite']
    # `suite_name` being the primary key of suite info
    suite_info = get_registered_suite_info(suite_name)
    runner_cls = suite_info['runner']
    runner_root_path = suite_info['runner_root_path']
    return runner_cls(runner_root_path=runner_root_path)


def build_model_from_model_info(model_info, config=None):
    suite_name = model_info['suite']
    # `suite_name` being the primary key of suite info
    suite_info = get_registered_suite_info(suite_name)
    model_cls = suite_info['model']
    model_name = model_info['model_name']
    return model_cls(model_name=model_name, config=config)


MODEL_INFO_REQUIRED_KEYS = ('model_name', 'suite', 'config_path',
                            'auto_compression_config_path', 'supported_apis')
MODEL_INFO_PRIMARY_KEY = 'model_name'
MODEL_INFO_REGISTRY = Registry(MODEL_INFO_REQUIRED_KEYS,
                               MODEL_INFO_PRIMARY_KEY)

SUITE_INFO_REQUIRED_KEYS = ('suite_name', 'model', 'runner', 'config',
                            'dataset_checker', 'runner_root_path')
SUITE_INFO_PRIMARY_KEY = 'suite_name'
SUITE_INFO_REGISTRY = Registry(SUITE_INFO_REQUIRED_KEYS,
                               SUITE_INFO_PRIMARY_KEY)

# Relations:
# 'suite' in model info <-> 'suite_name' in suite info

# Set aliases for backward compatibility
model_zoo = MODEL_INFO_REGISTRY
suite_zoo = SUITE_INFO_REGISTRY
get_registered_model_info = MODEL_INFO_REGISTRY.query
get_registered_suite_info = SUITE_INFO_REGISTRY.query
register_model_info = MODEL_INFO_REGISTRY.register_record
register_suite_info = SUITE_INFO_REGISTRY.register_record
