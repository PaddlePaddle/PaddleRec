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

import abc
import inspect
import functools

from .config import Config
from .register import (get_registered_model_info, build_runner_from_model_info,
                       build_model_from_model_info)
from .utils.misc import CachedProperty as cached_property
from .utils.cache import create_yaml_config_file
from .utils.logging import warn


class PaddleModel(object):
    # We constrain function params here
    def __new__(cls, model_name=None, config=None):
        if model_name is None and config is None:
            raise ValueError(
                "At least one of `model_name` and `config` must be not None.")
        elif model_name is not None and config is not None:
            if model_name != config.model_name:
                raise ValueError(
                    "If both `model_name` and `config` are not None, `model_name` should be the same as `config.model_name`."
                )
        elif model_name is None and config is not None:
            model_name = config.model_name
        model_info = get_registered_model_info(model_name)
        return build_model_from_model_info(
            model_info=model_info, config=config)


class BaseModel(metaclass=abc.ABCMeta):
    """
    Abstract base class of Model.
    
    Model defines how Config and Runner interact with each other. In addition, Model 
    provides users with multiple APIs to perform model training, prediction, etc.

    Args:
        model_name (str): A registered model name.
        config (config.BaseConfig|None, optional): Config. Default: None.
    """

    _API_FULL_LIST = ('train', 'evaluate', 'predict', 'export', 'infer',
                      'compression')
    _API_SUPPORTED_OPTS_KEY_PATTERN = 'supported_{api_name}_opts'

    def __init__(self, model_name, config=None):
        super().__init__()

        self.name = model_name
        self.model_info = get_registered_model_info(model_name)
        # NOTE: We build runner instance here by extracting runner info from model info
        # so that we don't have to overwrite the `__init__()` method of each child class.
        self.runner = build_runner_from_model_info(self.model_info)
        if config is None:
            warn(
                "We strongly discourage leaving `config` unset or setting it to None. "
                "Please note that when `config` is None, default settings will be used for every unspecified configuration item, "
                "which may lead to unexpected result. Please make sure that this is what you intend to do."
            )
            config = Config(model_name)
        self.config = config

        self._patch_apis()

    @abc.abstractmethod
    def train(self,
              dataset=None,
              batch_size=None,
              learning_rate=None,
              epochs_iters=None,
              ips=None,
              device='gpu',
              resume_path=None,
              dy2st=False,
              amp='OFF',
              use_vdl=True,
              save_dir=None):
        """
        Train a model.

        Args:
            dataset (str|None): Root path of the dataset. If None, use the setting in the config file or a 
                pre-defined default dataset.
            batch_size (int|None): Number of samples in each mini-batch. If multiple devices are used, this
                is the batch size on each device. If None, use the setting in the config file or a 
                pre-defined default batch size.
            learning_rate (float|None): Learning rate of model training. If None, use the setting in the config
                file.
            epochs_iters (int|None): Total epochs or iterations of model training. If None, use the setting in
                the config file or a pre-defined default value of epochs/iterations.
            ips (str|None): If not None, enable multi-machine training mode. `ips` specifies Paddle cluster node 
                ips, e.g., '192.168.0.16,192.168.0.17'.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'gpu', 'gpu:1,2'. 
                Default: 'gpu'.
            resume_path (str|None): If not None, resume training from the model snapshot stored in `resume_path`.
                If None, use the setting in the config file or a default setting.
            dy2st (bool): Whether or not to enable dynamic-to-static training. Default: False.
            amp (str): Optimization level to use in AMP training. Choices are ['O1', 'O2', 'OFF']. Default: 'OFF'.
            use_vdl (bool): Whether or not to enable VisualDL during training. Default: True.
            save_dir (str|None): Directory to store model snapshots and logs. If None, use the setting in the
                config file.

        Returns:
            subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self,
                 weight_path,
                 dataset=None,
                 batch_size=None,
                 ips=None,
                 device='gpu',
                 amp='OFF'):
        """
        Evaluate a model.

        Args:
            weight_path (str): Path of the weights to initialize the model.
            dataset (str|None): Root path of the dataset. If None, use the setting in the config file or a 
                pre-defined default dataset.
            batch_size (int|None): Number of samples in each mini-batch. If multiple devices are used, this
                is the batch size on each device. If None, use the setting in the config file or a 
                pre-defined default batch size.
            ips (str|None): If not None, enable multi-machine evaluation mode. `ips` specifies Paddle cluster 
                node ips, e.g., '192.168.0.16,192.168.0.17'.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'gpu', 'gpu:1,2'. 
                Default: 'gpu'.
            amp (str): Optimization level to use in AMP training. Choices are ['O1', 'O2', 'OFF']. Default: 'OFF'.

        Returns:
            subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, weight_path, input_path, device='gpu', save_dir=None):
        """
        Make prediction with a pre-trained model.

        Args:
            weight_path (str): Path of the weights to initialize the model.
            input_path (str): Path of the input file, e.g. an image.
            device (str): A string that describes the device to use, e.g., 'cpu', 'gpu'. Default: 'gpu'.
            save_dir (str|None): Directory to store prediction results. If None, use the setting in the config 
                file.

        Returns:
            subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, weight_path, save_dir):
        """
        Export a pre-trained model.

        Args:
            weight_path (str): Path of the weights to initialize the model.
            save_dir (str): Directory to store the exported model. 
        
        Returns:
            subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, model_dir, input_path, device='gpu', save_dir=None):
        """
        Make inference with an exported inference model.

        Args:
            model_dir (str): Path of the model snapshot to load.
            input_path (str): Path of the input file, e.g. an image.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'gpu'. Default: 'gpu'.
            save_dir (str|None): Directory to store inference results. If None, use the setting in the config 
                file.

        Returns:
            subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compression(self,
                    weight_path,
                    dataset=None,
                    batch_size=None,
                    learning_rate=None,
                    epochs_iters=None,
                    device='gpu',
                    use_vdl=True,
                    save_dir=None):
        """
        Perform quantization aware training (QAT) and export the quantized model.

        Args:
            weight_path (str): Path of the weights to initialize the model.
            dataset (str|None): Root path of the dataset. If None, use the setting in the config file or a 
                pre-defined default dataset.
            batch_size (int|None): Number of samples in each mini-batch. If multiple devices are used, this
                is the batch size on each device. If None, use the setting in the config file or a 
                pre-defined default batch size.
            learning_rate (float|None): Learning rate of qat training. If None, use the setting in the config
                file.
            epochs_iters (int|None): Total epochs of iterations of model training. If None, use the setting in
                the config file or a pre-defined default value of epochs/iterations.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'gpu'. Default: 'gpu'.
            use_vdl (bool): Whether or not to enable VisualDL during training. Default: True.
            save_dir (str|None): Directory to store inference results. If None, use the setting in the config 
                file.

        Returns:
            tuple[subprocess.CompletedProcess]
        """
        raise NotImplementedError

    @cached_property
    def _config_path(self):
        # NOTE: If multiple APIs share the same config path (e.g. `self._config_path`),
        # they should be called in a blocking manner.
        cls = self.__class__
        model_name = self.model_info['model_name']
        tag = '_'.join([cls.__name__.lower(), model_name])
        # Allow overwriting
        return create_yaml_config_file(tag=tag, noclobber=False)

    @cached_property
    def supported_apis(self):
        return self.model_info.get('supported_apis', None)

    @cached_property
    def supported_train_opts(self):
        return self.model_info.get(
            self._API_SUPPORTED_OPTS_KEY_PATTERN.format(api_name='train'),
            None)

    @cached_property
    def supported_evaluate_opts(self):
        return self.model_info.get(
            self._API_SUPPORTED_OPTS_KEY_PATTERN.format(api_name='evaluate'),
            None)

    @cached_property
    def supported_predict_opts(self):
        return self.model_info.get(self._API_SUPPORTED_OPTS_KEY_PATTERN.format(
            api_name='predict'),
                                   None)

    @cached_property
    def supported_infer_opts(self):
        return self.model_info.get(
            self._API_SUPPORTED_OPTS_KEY_PATTERN.format(api_name='infer'),
            None)

    @cached_property
    def supported_compression_opts(self):
        return self.model_info.get(self._API_SUPPORTED_OPTS_KEY_PATTERN.format(
            api_name='compression'),
                                   None)

    @cached_property
    def supported_dataset_types(self):
        return self.model_info.get('supported_dataset_types', None)

    def _patch_apis(self):
        def _make_unavailable(bnd_method):
            @functools.wraps(bnd_method)
            def _unavailable_api(*args, **kwargs):
                model_name = self.name
                api_name = bnd_method.__name__
                raise RuntimeError(
                    f"{model_name} does not support `{api_name}()`.")

            return _unavailable_api

        def _add_prechecks(bnd_method):
            @functools.wraps(bnd_method)
            def _api_with_prechecks(*args, **kwargs):
                sig = inspect.Signature.from_callable(bnd_method)
                bnd_args = sig.bind(*args, **kwargs)
                args_dict = bnd_args.arguments
                # Merge default values
                for p in sig.parameters.values():
                    if p.name not in args_dict and p.default is not p.empty:
                        args_dict[p.name] = p.default

                # Rely on nonlocal variable `checks`
                for check in checks:
                    try:
                        check.check(args_dict)
                    except _CheckFailed as e:
                        raise RuntimeError(
                            f"Unsupported options are found when calling `{api_name}()`: \n  {str(e)}"
                        )

                return bnd_method(*args, **kwargs)

            api_name = bnd_method.__name__
            checks = []
            # We hardcode the prechecks for each API here
            if api_name == 'train':
                opts = self.supported_train_opts
                if opts is not None:
                    if 'device' in opts:
                        checks.append(
                            _CheckDevice(
                                opts['device'],
                                self.runner.parse_device,
                                check_mc=True))
                    if 'dy2st' in opts:
                        checks.append(_CheckDy2St(opts['dy2st']))
                    if 'amp' in opts:
                        checks.append(_CheckAMP(opts['amp']))
            elif api_name == 'evaluate':
                opts = self.supported_evaluate_opts
                if opts is not None:
                    if 'device' in opts:
                        checks.append(
                            _CheckDevice(
                                opts['device'],
                                self.runner.parse_device,
                                check_mc=True))
                    if 'amp' in opts:
                        checks.append(_CheckAMP(opts['amp']))
            elif api_name == 'predict':
                opts = self.supported_predict_opts
                if opts is not None:
                    if 'device' in opts:
                        checks.append(
                            _CheckDevice(
                                opts['device'],
                                self.runner.parse_device,
                                check_mc=False))
            elif api_name == 'infer':
                opts = self.supported_infer_opts
                if opts is not None:
                    if 'device' in opts:
                        checks.append(
                            _CheckDevice(
                                opts['device'],
                                self.runner.parse_device,
                                check_mc=False))
            elif api_name == 'compression':
                opts = self.supported_compression_opts
                if opts is not None:
                    if 'device' in opts:
                        checks.append(
                            _CheckDevice(
                                opts['device'],
                                self.runner.parse_device,
                                check_mc=True))
            else:
                return bnd_method

            return _api_with_prechecks

        supported_apis = self.supported_apis
        if supported_apis is not None:
            avail_api_set = set(self.supported_apis)
        else:
            avail_api_set = set(self._API_FULL_LIST)
        for api_name in self._API_FULL_LIST:
            api = getattr(self, api_name)
            if api_name not in avail_api_set:
                # We decorate old API implementation with `_make_unavailable` 
                # so that an error is always raised when the API is called.
                decorated_api = _make_unavailable(api)
            else:
                # We decorate old API implementation with `_add_prechecks` to 
                # perform validity checks before invoking the internal API.
                decorated_api = _add_prechecks(api)

            # Monkey-patch
            setattr(self, api_name, decorated_api)


class _CheckFailed(Exception):
    def __init__(self, arg_name, arg_val, legal_vals):
        self.arg_name = arg_name
        self.arg_val = arg_val
        self.legal_vals = legal_vals

    def __str__(self):
        return f"`{self.arg_name}` is expected to be one of or conforms to {self.legal_vals}, but got {self.arg_val}"


class _APICallArgsChecker(object):
    def __init__(self, legal_vals):
        super().__init__()
        self.legal_vals = legal_vals

    def check(self, args):
        raise NotImplementedError


class _CheckDevice(_APICallArgsChecker):
    def __init__(self, legal_vals, parse_device, check_mc=False):
        super().__init__(legal_vals)
        self.parse_device = parse_device
        self.check_mc = check_mc

    def check(self, args):
        assert 'device' in args
        device = args['device']
        if device is not None:
            device_type, dev_ids = self.parse_device(device)
            if not self.check_mc:
                if device_type not in self.legal_vals:
                    raise _CheckFailed('device', device, self.legal_vals)
            else:
                # Currently we only check multi-device settings for GPUs
                if device_type != 'gpu':
                    if device_type not in self.legal_vals:
                        raise _CheckFailed('device', device, self.legal_vals)
                else:
                    n1c1_desc = f'{device_type}_n1c1'
                    n1cx_desc = f'{device_type}_n1cx'
                    nxcx_desc = f'{device_type}_nxcx'

                    if len(dev_ids) <= 1:
                        if (n1c1_desc not in self.legal_vals and
                                n1cx_desc not in self.legal_vals and
                                nxcx_desc not in self.legal_vals):
                            raise _CheckFailed('device', device,
                                               self.legal_vals)
                    else:
                        assert 'ips' in args
                        if args['ips'] is not None:
                            # Multi-machine
                            if nxcx_desc not in self.legal_vals:
                                raise _CheckFailed('device', device,
                                                   self.legal_vals)
                        else:
                            # Single-machine multi-device
                            if (n1cx_desc not in self.legal_vals and
                                    nxcx_desc not in self.legal_vals):
                                raise _CheckFailed('device', device,
                                                   self.legal_vals)
        else:
            # When `device` is None, we assume that a default device that the 
            # current model supports will be used, so we simply do nothing.
            pass


class _CheckDy2St(_APICallArgsChecker):
    def check(self, args):
        assert 'dy2st' in args
        dy2st = args['dy2st']
        if isinstance(self.legal_vals, list):
            assert len(self.legal_vals) == 1
            support_dy2st = bool(self.legal_vals[0])
        else:
            support_dy2st = bool(self.legal_vals)
        if dy2st is not None:
            if dy2st and not support_dy2st:
                raise _CheckFailed('dy2st', dy2st, [support_dy2st])
        else:
            pass


class _CheckAMP(_APICallArgsChecker):
    def check(self, args):
        assert 'amp' in args
        amp = args['amp']
        if amp is not None:
            if amp != 'OFF' and amp not in self.legal_vals:
                raise _CheckFailed('amp', amp, self.legal_vals)
        else:
            pass
