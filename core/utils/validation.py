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

from paddlerec.core.utils import envs


class ValueFormat:
    def __init__(self, value_type, value, value_handler, required=False):
        self.value_type = value_type
        self.value_handler = value_handler
        self.value = value
        self.required = required

    def is_valid(self, name, value):

        if not self.value_type:
            ret = True
        else:
            ret = self.is_type_valid(name, value)

        if not ret:
            return ret

        if not self.value or not self.value_handler:
            return True

        ret = self.is_value_valid(name, value)
        return ret

    def is_type_valid(self, name, value):
        if self.value_type == "int":
            if not isinstance(value, int):
                print("\nattr {} should be int, but {} now\n".format(
                    name, type(value)))
                return False
            return True

        elif self.value_type == "str":
            if not isinstance(value, str):
                print("\nattr {} should be str, but {} now\n".format(
                    name, type(value)))
                return False
            return True

        elif self.value_type == "strs":
            if not isinstance(value, list):
                print("\nattr {} should be list(str), but {} now\n".format(
                    name, type(value)))
                return False
            for v in value:
                if not isinstance(v, str):
                    print("\nattr {} should be list(str), but list({}) now\n".
                          format(name, type(v)))
                    return False
            return True

        elif self.value_type == "dict":
            if not isinstance(value, dict):
                print("\nattr {} should be str, but {} now\n".format(
                    name, type(value)))
                return False
            return True

        elif self.value_type == "dicts":
            if not isinstance(value, list):
                print("\nattr {} should be list(dist), but {} now\n".format(
                    name, type(value)))
                return False
            for v in value:
                if not isinstance(v, dict):
                    print("\nattr {} should be list(dist), but list({}) now\n".
                          format(name, type(v)))
                    return False
            return True

        elif self.value_type == "ints":
            if not isinstance(value, list):
                print("\nattr {} should be list(int), but {} now\n".format(
                    name, type(value)))
                return False
            for v in value:
                if not isinstance(v, int):
                    print("\nattr {} should be list(int), but list({}) now\n".
                          format(name, type(v)))
                    return False
            return True

        else:
            print("\nattr {}'s type is {}, can not be supported now\n".format(
                name, type(value)))
            return False

    def is_value_valid(self, name, value):
        ret = self.value_handler(name, value, self.value)
        return ret


def in_value_handler(name, value, values):
    if value not in values:
        print("\nattr {}'s value is {}, but {} is expected\n".format(
            name, value, values))
        return False
    return True


def eq_value_handler(name, value, values):
    if value != values:
        print("\nattr {}'s value is {}, but == {} is expected\n".format(
            name, value, values))
        return False
    return True


def ge_value_handler(name, value, values):
    if value < values:
        print("\nattr {}'s value is {}, but >= {} is expected\n".format(
            name, value, values))
        return False
    return True


def le_value_handler(name, value, values):
    if value > values:
        print("\nattr {}'s value is {}, but <= {} is expected\n".format(
            name, value, values))
        return False
    return True


def register():
    validations = {}
    validations["workspace"] = ValueFormat("str", None, None, True)
    validations["mode"] = ValueFormat(None, None, None, True)
    validations["runner"] = ValueFormat("dicts", None, None, True)
    validations["phase"] = ValueFormat("dicts", None, None, True)
    validations["hyper_parameters"] = ValueFormat("dict", None, None, False)
    return validations


def yaml_validation(config):
    all_checkers = register()

    require_checkers = []
    for name, checker in all_checkers.items():
        if checker.required:
            require_checkers.append(name)

    _config = envs.load_yaml(config)

    for required in require_checkers:
        if required not in _config.keys():
            print("\ncan not find {} in yaml, which is required\n".format(
                required))
            return False

    for name, value in _config.items():
        checker = all_checkers.get(name, None)
        if checker:
            ret = checker.is_valid(name, value)
            if not ret:
                return False

    return True
