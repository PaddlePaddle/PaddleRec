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

from .cache import persist

RES_DICT_TYPE = dict


def build_res_dict(res_flag, err_type=None, err_msg=None, **kwargs):
    if res_flag:
        if err_type is not None:
            raise ValueError(
                f"`res_flag` is {res_flag}, but `err_type` is not None.")
        if err_msg is not None:
            raise ValueError(
                f"`res_flag` is {res_flag}, but `err_msg` is not None.")
        return RES_DICT_TYPE(res_flag=res_flag, **kwargs)
    else:
        if err_type is None:
            raise ValueError(
                f"`res_flag` is {res_flag}, but `err_type` is None.")
        if err_msg is None:
            if _is_known_error_type(err_type):
                err_msg = f""
            else:
                raise ValueError(
                    f"{err_type} is not a known error type, in which case `err_msg` must be specified to a value other than None."
                )
        return RES_DICT_TYPE(
            res_flag=res_flag, err_type=err_type, err_msg=err_msg, **kwargs)


def _is_dataset_valid(res_dict):
    assert isinstance(res_dict, RES_DICT_TYPE)
    flag = res_dict['res_flag']
    return flag


def _is_known_error_type(err_type):
    return isinstance(err_type, CheckFailedError)


# TODO: Should we move the exception definitions to a separate file?


class CheckFailedError(Exception):
    def __init__(self, err_info=None, solution=None, message=None):
        if message is None:
            message = self._construct_message(err_info, solution)
        super().__init__(message)

    def _construct_message(self, err_info, solution):
        if err_info is None:
            return ""
        else:
            msg = f"Dataset check failed. We encountered the following error:\n  {err_info}"
            if solution is not None:
                msg += f"\nPlease try to resolve the issue as follows:\n  {solution}"
            return msg


class UnsupportedDatasetTypeError(CheckFailedError):
    def __init__(self,
                 dataset_type=None,
                 err_info=None,
                 solution=None,
                 message=None):
        if err_info is None:
            if dataset_type is not None:
                err_info = f"{repr(dataset_type)} is not a supported dataset type."
        super().__init__(err_info, solution, message)


class DatasetFileNotFoundError(CheckFailedError):
    def __init__(self,
                 file_path=None,
                 err_info=None,
                 solution=None,
                 message=None):
        if err_info is None:
            if file_path is not None:
                err_info = f"`{file_path}` does not exist."
        super().__init__(err_info, solution, message)


persist_dataset_meta = persist(cond=_is_dataset_valid)
