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
"""helper for config
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
from collections import defaultdict

from pgl.utils.logger import log


class AttrDict(dict):
    """ attr dict  """

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
        """ set attr """
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(AttrDict, self).__setattr__(name, value)
        super(AttrDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, attr):
        """ get attr """
        try:
            value = super(AttrDict, self).__getitem__(attr)
        except KeyError:
            #  log.warn("%s attribute is not existed, return None" % attr)
            warnings.warn("%s attribute is not existed, return None" % attr)
            value = None
        return value

    def update(self, e=None, **f):
        """ update """
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        """ pop """
        delattr(self, k)
        return super(AttrDict, self).pop(k, d)


def make_dir(path):
    """Build directory"""
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(config_file):
    """Load config file"""
    with open(config_file) as f:
        if hasattr(yaml, 'FullLoader'):
            config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            config = yaml.load(f)
    config = AttrDict(config)
    return config


def get_all_edge_type(etype2files, symmetry):
    """ get all edge type """
    if symmetry:
        etype_list = []
        for etype in etype2files.keys():
            r_etype = get_inverse_etype(etype)
            etype_list.append(etype)
            if r_etype != etype:
                etype_list.append(r_etype)
    else:
        etype_list = list(etype2files.keys())

    return etype_list


def get_inverse_etype(etype):
    """ get inverse type """
    fields = etype.split("2")
    if len(fields) == 3:
        src, etype, dst = fields
        r_etype = "2".join([dst, etype, src])
    else:
        r_etype = "2".join([fields[1], fields[0]])
    return r_etype


def parse_files(type_files):
    """ parse files """
    type2files = {}
    for item in type_files.split(","):
        t, file_or_dir = item.split(":")
        type2files[t] = file_or_dir
    return type2files


def generate_files_string(ntype_list, nodes_file):
    """ generate files string """
    res = ""
    for item in ntype_list:
        res += item + ":" + nodes_file + ","
    return res[:-1]


def get_files(edge_file_or_dir):
    """ get files """
    if os.path.isdir(edge_file_or_dir):
        ret_files = []
        files = glob.glob(os.path.join(edge_file_or_dir, "*"))
        for file_ in files:
            if os.path.isdir(file_):
                log.info("%s is a directory, not a file" % file_)
            else:
                ret_files.append(file_)
    elif "*" in edge_file_or_dir:
        ret_files = []
        files = glob.glob(edge_file_or_dir)
        for file_ in files:
            if os.path.isdir(file_):
                log.info("%s is a directory, not a file" % file_)
            else:
                ret_files.append(file_)
    else:
        ret_files = [edge_file_or_dir]
    return ret_files


def load_ip_addr(ip_config):
    """ load ip addr """
    ip_addr_list = []
    with open(ip_config, 'r') as f:
        for line in f:
            ip_addr_list.append(line.strip())
    ip_addr = ";".join(ip_addr_list)
    return ip_addr


def convert_nfeat_info(nfeat_info):
    """ convert node feat info """
    res = defaultdict(dict)
    for item in nfeat_info:
        res[item[0]].update({item[1]: [item[2], item[3]]})
    return res


def make_nfeat_info(ntype_list):
    """ make node feat info """
    res = []
    for ntype in ntype_list:
        res.append([ntype, "s", "string", 1])
    return res
