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

import inspect
import threading

from . import logging
from ..flags import DEBUG


class _Singleton(type):
    _insts = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._insts:
            with cls._lock:
                if cls not in cls._insts:
                    cls._insts[cls] = super().__call__(*args, **kwargs)
        return cls._insts[cls]


class _EMPTY_ARG(metaclass=_Singleton):
    def __repr__(self):
        return "-EMPTY-"


# TODO: Replacing the code that forwards arguments using `locals()` with more explicit code


def running_datacheck(data_path, data_type, yaml_path=_EMPTY_ARG()):
    """
    Args:
        data_path (str): Root path of the dataset.
        data_type (str): Type of the dataset.
        yaml_path (str, optinoal): Absolute path of the YAML file of the dataset.
    """

    return _stagelog_call('running_datacheck', **locals())


def running_train(learning_rate, epoch_iters, batch_size, data_path, yaml_path,
                  vdl_path, save_dir):
    """
    Args:
        learning_rate (float): Learning rate of model training.
        epoch_iters (int): Total epochs or iterations of model training.
        batch_size (int): Number of samples in each mini-batch.
        data_path (str): Root path of the dataset used in training.
        yaml_path (str): Absolute path of the YAML configuration file.
        vdl_path (str): Path of VisualDL `logdir`.
        save_dir (str): Directory that contains model snapshots and logs.
    """

    return _stagelog_call('running_train', **locals())


def running_verify(checkpoint_dir,
                   checkpoint_name,
                   metrics,
                   save_dir=_EMPTY_ARG()):
    """
    Args:
        checkpoint_dir (str): Directory that contains the weights to initialize the model.
        checkpoint_name (str): Base name of the weight file.
        metrics (list[str]): A list of names of all metrics used in model evaluation.
        save_dir (str, optional): Directory that contains model snapshots and logs.
    """
    return _stagelog_call('running_verify', **locals())


def running_compress(checkpoint_dir,
                     checkpoint_name,
                     compress_batch_size,
                     compress_epoch,
                     compress_learning_rate,
                     vdl_path,
                     save_dir,
                     metrics=_EMPTY_ARG()):
    """
    Args:
        checkpoint_dir (str): Directory that contains the weights to initialize the model.
        checkpoint_name (str): Base name of the weight file.
        compress_batch_size (int): Number of samples in each mini-batch.
        compress_epoch (int): Total epochs or iterations of QAT.
        compress_learning_rate (float): Learning rate of model training.
        vdl_path (str): Path of VisualDL `logdir`.
        save_dir (str): Directory that contains model snapshots and logs.
        metrics (list[str], optional): A list of names of all metrics used in model evaluation.
    """

    return _stagelog_call('running_compress', **locals())


def running_deploy(operating_system, language, architecture, accelerator):
    return _stagelog_call('running_deploy', **locals())


def success(stage_id, result=_EMPTY_ARG()):
    return _stagelog_call('success', **locals())


def fail(stage_id, error_message=_EMPTY_ARG()):
    return _stagelog_call('fail', **locals())


def success_datacheck(stage_id, train_dataset, validation_dataset,
                      test_dataset):
    """
    Args:
        stage_id (str): ID of current stage.
        train_dataset (int): Number of samples in training dataset.
        validation_dataset (int): Number of samples in validation dataset.
        test_dataset (int): Number of samples in test dataset.
    """
    return _stagelog_call('success_datacheck', **locals())


def _stagelog_call(func_name, *args, **kwargs):
    try:
        # Lazy import
        import stagelog
    except ModuleNotFoundError:
        if DEBUG:
            logging.warn(
                "`stagelog` is not found. Please check if it is properly installed."
            )
        else:
            pass
    else:
        func = getattr(stagelog, func_name)
        # Ignore optional arguments with empty values
        empty = _EMPTY_ARG()
        args = [arg for arg in args if arg is not empty]
        kwargs = {k: v for k, v in kwargs.items() if v is not empty}

        try:
            return func(*args, **kwargs)
        except stagelog.exception.RecorderNotInitException:
            if DEBUG:
                logging.warn("stagelog not initialized.")
            else:
                pass


class _StageLogContextManagerMeta(type):
    def __new__(mcls, name, bases, attrs):
        cls = super().__new__(mcls, name, bases, attrs)
        stagelog_api = getattr(cls, '_STAGELOG_API')
        if stagelog_api is not None:
            sig = inspect.signature(stagelog_api)
            cls.__init__.__signature__ = sig.replace(
                parameters=[
                    inspect.Parameter(
                        name='self',
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    *sig.parameters.values()
                ],
                return_annotation=sig.empty)
            cls.__init__.__doc__ = stagelog_api.__doc__
        return cls


class _StageLogContextManager(metaclass=_StageLogContextManagerMeta):
    _STAGELOG_API = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.running_args = args
        self.running_kwargs = kwargs

    def __enter__(self):
        self.stage_id = self._log_running()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._log_status(self.stage_id, exc_type, exc_val, exc_tb)

    def _log_running(self):
        return type(self)._STAGELOG_API(*self.running_args,
                                        **self.running_kwargs)

    def _log_status(self, stage_id, exc_type, exc_val, exc_tb):
        raise NotImplementedError


class StageLogTrain(_StageLogContextManager):
    _STAGELOG_API = running_train

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _log_status(self, stage_id, exc_type, exc_val, exc_tb):
        if exc_type is None and exc_val is None and exc_tb is None:
            success(stage_id)
        else:
            fail(stage_id, str(exc_val))


class StageLogCompress(_StageLogContextManager):
    _STAGELOG_API = running_compress

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _log_status(self, stage_id, exc_type, exc_val, exc_tb):
        if exc_type is None and exc_val is None and exc_tb is None:
            success(stage_id)
        else:
            fail(stage_id, str(exc_val))


class StageLogEvaluate(_StageLogContextManager):
    _STAGELOG_API = running_verify

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_eval_ret(self, eval_ret):
        if not hasattr(eval_ret, 'returncode'):
            raise TypeError("`eval_ret` must have attribute `returncode`.")
        self.eval_ret = eval_ret

    def _log_status(self, stage_id, exc_type, exc_val, exc_tb):
        if exc_type is None and exc_val is None and exc_tb is None:
            if not hasattr(self, 'eval_ret'):
                raise RuntimeError("`eval_ret` is not set.")
            cp = self.eval_ret
            if cp.returncode == 0:
                success(stage_id, result=cp.metrics)
            else:
                # Since `cp.stderr` can be very long, currently we hard-code 
                # the error message.
                fail(stage_id, "Model evaluation failed.")
        else:
            return False
