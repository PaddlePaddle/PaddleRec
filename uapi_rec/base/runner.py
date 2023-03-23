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
import sys
import abc
import io
import locale
import subprocess
import asyncio

from .utils.misc import run_cmd as _run_cmd, abspath


class BaseRunner(metaclass=abc.ABCMeta):
    """
    Abstract base class of Runner.

    Runner is responsible for executing training/inference/compression commands.

    Args:
        runner_root_path (str): Path of the directory where the scripts reside.
    """

    def __init__(self, runner_root_path):
        super().__init__()

        self.runner_root_path = abspath(runner_root_path)
        # Path to python interpreter
        self.python = sys.executable

    def prepare(self):
        """
        Make preparations for the execution of commands.

        For example, download prerequisites and install dependencies.
        """
        # By default we do nothing
        pass

    @abc.abstractmethod
    def train(self, config_path, cli_args, device, ips):
        """
        Execute model training command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[utils.arg.CLIArgument]): List of command-line Arguments.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'.
            ips (str): Paddle cluster node ips, e.g., '192.168.0.16,192.168.0.17'.

        Returns:
            subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, config_path, cli_args, device, ips):
        """
        Execute model evaluation command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[utils.arg.CLIArgument]): List of command-line Arguments.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'.
            ips (str): Paddle cluster node ips, e.g., '192.168.0.16,192.168.0.17'.

        Returns:
            subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, config_path, cli_args, device):
        """
        Execute prediction command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[utils.arg.CLIArgument]): List of command-line Arguments.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'.

        Returns:
            subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, config_path, cli_args, device):
        """
        Execute model export command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[utils.arg.CLIArgument]): List of command-line Arguments.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'.

        Returns:
            subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, config_path, cli_args, device):
        """
        Execute model inference command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[utils.arg.CLIArgument]): List of command-line Arguments.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'.

        Returns:
            subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compression(self, config_path, train_cli_args, export_cli_args, device,
                    train_save_dir):
        """
        Execute model compression (quantization aware training and model export) commands.

        Args:
            config_path (str): Path of the configuration file.
            train_cli_args (list[utils.arg.CLIArgument]): List of command-line Arguments used for model 
                training.
            train_cli_args (list[utils.arg.CLIArgument]): List of command-line Arguments used for model 
                export.
            device (str): A string that describes the device(s) to use, e.g., 'cpu', 'xpu:0', 'gpu:1,2'.
            train_save_dir (str): Directory to store model snapshots and the exported model.

        Returns:
            tuple[subprocess.CompletedProcess]
        """
        raise NotImplementedError

    def distributed(self, device, ips=None):
        # TODO: docstring
        python = self.python
        if device is None:
            # By default use a GPU device
            return python, 'gpu'
        device, dev_ids = self.parse_device(device)
        if len(dev_ids) == 0:
            return python
        else:
            num_devices = len(dev_ids)
            dev_ids = ','.join(dev_ids)
        if num_devices > 1:
            python += " -m paddle.distributed.launch"
            python += f" --gpus {dev_ids}"
            if ips is not None:
                python += f" --ips {ips}"
        elif num_devices == 1:
            # TODO: Accommodate Windows system
            python = f"CUDA_VISIBLE_DEVICES={dev_ids} {python}"
        return python

    def parse_device(self, device):
        # According to https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html
        if ':' not in device:
            return device, []
        else:
            device_type, dev_ids = device.split(':')
            dev_ids = dev_ids.split(',')
            return device_type, dev_ids

    def run_cmd(self,
                cmd,
                switch_wdir=True,
                silent=False,
                echo=True,
                pipe_stdout=False,
                pipe_stderr=False,
                blocking=True):
        if switch_wdir:
            if isinstance(switch_wdir, str):
                # In this case `switch_wdir` specifies a relative path
                cwd = os.path.join(self.runner_root_path, switch_wdir)
            else:
                cwd = self.runner_root_path
        else:
            cwd = None

        if blocking:
            return _run_cmd(
                cmd,
                cwd=cwd,
                silent=silent,
                echo=echo,
                pipe_stdout=pipe_stdout,
                pipe_stderr=pipe_stderr,
                blocking=blocking)
        else:
            # Refer to 
            # https://stackoverflow.com/questions/17190221/subprocess-popen-cloning-stdout-and-stderr-both-to-terminal-and-variables/25960956
            @asyncio.coroutine
            def _read_display_and_record_from_stream(in_stream, out_stream,
                                                     buf):
                # According to
                # https://docs.python.org/3/library/subprocess.html#frequently-used-arguments
                _ENCODING = locale.getpreferredencoding(False)
                chars = []
                out_stream_is_buffered = hasattr(out_stream, 'buffer')
                while True:
                    flush = False
                    char = yield from in_stream.read(1)
                    if char == b'':
                        break
                    if out_stream_is_buffered:
                        out_stream.buffer.write(char)
                    chars.append(char)
                    if char == b'\n':
                        flush = True
                    elif char == b'\r':
                        # NOTE: In order to get tqdm progress bars to produce normal outputs
                        # we treat '\r' as an ending character of line
                        flush = True
                    if flush:
                        line = b''.join(chars)
                        line = line.decode(_ENCODING)
                        if not out_stream_is_buffered:
                            # We use line buffering
                            out_stream.write(line)
                        else:
                            out_stream.buffer.flush()
                        buf.write(line)
                        chars.clear()

            @asyncio.coroutine
            def _tee_proc_call(proc_call, stdout_buf, stderr_buf):
                proc = yield from proc_call
                yield from asyncio.gather(
                    _read_display_and_record_from_stream(
                        proc.stdout, sys.stdout, stdout_buf),
                    _read_display_and_record_from_stream(
                        proc.stderr, sys.stderr, stderr_buf))
                # NOTE: https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait
                retcode = yield from proc.wait()
                return retcode

            if not (pipe_stdout and pipe_stderr):
                raise ValueError(
                    "In non-blocking mode, please set `pipe_stdout` and `pipe_stderr` to True."
                )

            # Non-blocking call with stdout and stderr piped
            with io.StringIO() as stdout_buf, io.StringIO() as stderr_buf:
                proc_call = _run_cmd(
                    cmd,
                    cwd=cwd,
                    echo=echo,
                    silent=silent,
                    pipe_stdout=True,
                    pipe_stderr=True,
                    blocking=False,
                    async_run=True)
                # FIXME: tqdm progress bars can not be normally displayed
                # XXX: For simplicity, we cache entire stdout and stderr content, which can 
                # take up lots of memory.
                loop = asyncio.get_event_loop()
                try:
                    retcode = loop.run_until_complete(
                        _tee_proc_call(proc_call, stdout_buf, stderr_buf))
                    cp = subprocess.CompletedProcess(cmd, retcode,
                                                     stdout_buf.getvalue(),
                                                     stderr_buf.getvalue())
                    return cp
                finally:
                    loop.close()
