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

import asyncio
import os
import subprocess

from .logging import info


def run_cmd(cmd,
            silent=True,
            cwd=None,
            timeout=None,
            echo=False,
            pipe_stdout=False,
            pipe_stderr=False,
            blocking=True,
            async_run=False,
            text=True):
    """Wrap around `subprocess.run()` to execute a shell command."""
    if blocking:
        cfg = dict(check=True, timeout=timeout, cwd=cwd)
    else:
        cfg = dict(cwd=cwd)

    if silent:
        cfg['stdout'] = subprocess.DEVNULL
    if not async_run and (pipe_stdout or pipe_stderr):
        cfg['text'] = True
    if pipe_stdout:
        cfg['stdout'] = subprocess.PIPE
    if pipe_stderr:
        cfg['stderr'] = subprocess.PIPE
        if blocking:
            cfg['check'] = False

    if echo:
        info(cmd)

    if blocking:
        # XXX: We run subprocess with `shell` set to True for ease of use.
        # However, we may have to explicitly consider shell types on different platforms.
        return subprocess.run(cmd, shell=True, **cfg)
    else:
        if async_run:
            return asyncio.create_subprocess_shell(cmd, **cfg)
        else:
            if text:
                cfg.update(dict(bufsize=1, text=True))
            else:
                cfg.update(dict(bufsize=0, text=False))
            return subprocess.Popen(cmd, shell=True, **cfg)


def abspath(path):
    return os.path.abspath(path)


class CachedProperty(object):
    """
    A property that is only computed once per instance and then replaces itself with an ordinary attribute.

    The implementation refers to https://github.com/pydanny/cached-property/blob/master/cached_property.py .
    
    Note that this implementation does NOT work in multi-thread or coroutine senarios.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func
        self.__doc__ = getattr(func, '__doc__', '')

    def __get__(self, obj, cls):
        if obj is None:
            return self
        val = self.func(obj)
        # Hack __dict__ of obj to inject the value
        # Note that this is only executed once
        obj.__dict__[self.func.__name__] = val
        return val
