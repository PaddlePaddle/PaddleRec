# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
""" Atrribute Dict
"""

import sys
import datetime
import os
import yaml
import random
import shutil
import six
import warnings
import glob


class AttrDict(dict):
    """ config dict """

    def __init__(self, d={}, **kwargs):
        """ init """
        if kwargs:
            d.update(**kwargs)

        for k, v in d.items():
            setattr(self, k, v)

        # Class attributes
        #  for k in self.__class__.__dict__.keys():
        #      if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #          setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        """ set config attr """
        if isinstance(value, (list, tuple)):
            value = [
                self.__class__(x) if isinstance(x, dict) else x for x in value
            ]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(AttrDict, self).__setattr__(name, value)
        super(AttrDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, attr):
        """ get config attr """
        try:
            value = super(AttrDict, self).__getitem__(attr)
        except KeyError:
            #  log.warn("%s attribute is not existed, return None" % attr)
            #  warnings.warn("%s attribute is not existed, return None" % attr)
            value = None
        return value

    def update(self, e=None, **f):
        """ update value """
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        """ pop attr """
        delattr(self, k)
        return super(AttrDict, self).pop(k, d)


def pretty(d, indent=0):
    """Pretty print config.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print('    ' * indent + "%s:" % str(key))
            pretty(value, indent + 1)
        else:
            print('    ' * indent + "%s: %s" % (str(key), repr(value)))


def get_last_dir(path):
    """Get the last directory of a path.
    """
    if os.path.isfile(path):
        # e.g: "../checkpoints/task_name/epoch0_step300/predict.txt"
        # return "epoch0_step300"
        last_dir = path.split("/")[-2]

    elif os.path.isdir(path):
        if path[-1] == '/':
            # e.g: "../checkpoints/task_name/epoch0_step300/"
            last_dir = path.split('/')[-2]
        else:
            # e.g: "../checkpoints/task_name/epoch0_step300"
            last_dir = path.split('/')[-1]
    else:
        # path or file is not existed
        warnings.warn('%s is not a existed file or path' % path)
        last_dir = ""

    return last_dir


def make_dir(path):
    """Build directory"""
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(config_file):
    """Load config file"""
    with open(config_file) as f:
        # remove comments
        lines = f.readlines()
        lines = filter(lambda x: not x.strip().startswith("#"), lines)
        lines = '\n'.join(lines)

        if hasattr(yaml, 'FullLoader'):
            config = yaml.load(lines, Loader=yaml.FullLoader)
        else:
            config = yaml.load(lines)
    return config


def create_necessary_dirs(config):
    """Create some necessary directories to save some important files.
    """
    config.log_dir = os.path.join(config.log_dir, config.task_name)
    config.save_dir = os.path.join(config.save_dir, config.task_name)
    config.output_dir = os.path.join(config.output_dir, config.task_name)

    make_dir(config.log_dir)
    make_dir(config.save_dir)
    make_dir(config.output_dir)


def save_files(config):
    """Save config file so that we can know the config when we look back
    """
    filelist = config.files2saved
    targetpath = config.log_dir

    if filelist is not None:
        for file_or_dir in filelist:
            if os.path.isdir(file_or_dir):
                last_name = get_last_dir(file_or_dir)
                dst = os.path.join(targetpath, last_name)
                try:
                    copy_and_overwrite(file_or_dir, dst)
                except Exception as e:
                    print(e)
                print("backup %s to %s" % (file_or_dir, targetpath))
            else:
                for filename in files(files=file_or_dir):
                    if os.path.isfile(filename):
                        print("backup %s to %s" % (filename, targetpath))
                        shutil.copy2(filename, targetpath)
                    else:
                        print("%s is not existed." % filename)


def copy_and_overwrite(from_path, to_path):
    """ copy and overwrite file """
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)


def files(curr_dir='./', files='*.py'):
    """ files """
    for i in glob.glob(os.path.join(curr_dir, files)):
        yield i


def prepare_config(config_file, isCreate=False, isSave=False):
    """ prepare config """
    if os.path.isfile(config_file):
        config = load_config(config_file)
        config = AttrDict(config)
    else:
        raise TypeError("%s is not a yaml file" % config_file)

    if isCreate:
        create_necessary_dirs(config)

    if isSave:
        save_files(config)

    return config
