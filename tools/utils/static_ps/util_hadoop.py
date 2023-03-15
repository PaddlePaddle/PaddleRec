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
"""Hadoop Utilities Functions
"""
import os
import sys
import json
import time
import math
import collections
import numpy as np

HADOOP_BIN = None
FS_NAME = None
FS_UGI = None
ERR_LOG = "./hadoop_err.log"
Ddfs = " -Ddfs.client.block.write.retries=15 -Ddfs.rpc.timeout=300000 -Ddfs.delete.trash=1"


def set_hadoop_account(hadoop_bin, fs_name, fs_ugi):
    """set hadoop account"""
    global HADOOP_BIN
    global FS_NAME
    global FS_UGI
    HADOOP_BIN = hadoop_bin
    FS_NAME = fs_name
    FS_UGI = fs_ugi


def set_hadoop_err(err_log="./hadoop_err.log"):
    """set hadoop err file"""
    global ERR_LOG
    ERR_LOG = err_log


def parse_account(hadoop_bin, fs_name, fs_ugi):
    """parse hadoop account"""
    is_local_account = not (hadoop_bin is None or fs_name is None or
                            fs_ugi is None)
    is_global_account = not (HADOOP_BIN is None or FS_NAME is None or
                             FS_UGI is None)

    if not is_local_account and not is_global_account:
        msg = "hadoop account should be setted before using hadoop commands." + \
            " But got [hadoop_bin = %s], [fs_name = %s] and [fs_ugi = %s]" % \
            (hadoop_bin, fs_name, fs_ugi)
        raise ValueError(msg)
    elif is_global_account:
        hadoop_bin = HADOOP_BIN
        fs_name = FS_NAME
        fs_ugi = FS_UGI

    return hadoop_bin, fs_name, fs_ugi


def check_hadoop_path(path, hadoop_bin=None, fs_name=None, fs_ugi=None):
    """check hadoop path"""
    hadoop_bin, fs_name, fs_ugi = parse_account(hadoop_bin, fs_name, fs_ugi)

    if path.startswith("hdfs://") or path.startswith("afs://"):
        return path
    else:
        real_path = fs_name + path
        return real_path


def make_base_cmd(hadoop_bin, fs_name, fs_ugi):
    """make base hadoop command"""
    cmd = "%s fs" % hadoop_bin
    cmd += " -D fs.default.name=%s" % fs_name
    cmd += " -D hadoop.job.ugi=%s" % fs_ugi
    cmd += Ddfs

    return cmd


def ls(path, hadoop_bin=None, fs_name=None, fs_ugi=None):
    """hadoop list"""
    hadoop_bin, fs_name, fs_ugi = parse_account(hadoop_bin, fs_name, fs_ugi)
    path = check_hadoop_path(path, hadoop_bin, fs_name, fs_ugi)
    cmd = make_base_cmd(hadoop_bin, fs_name, fs_ugi)

    cmd += " -ls %s" % path
    cmd += " | awk '{print $8}'"
    cmd += " 2>%s" % ERR_LOG
    filelist = os.popen(cmd).read().split()
    return filelist


def mkdir(path, hadoop_bin=None, fs_name=None, fs_ugi=None):
    """hadoop mkdir directory"""
    hadoop_bin, fs_name, fs_ugi = parse_account(hadoop_bin, fs_name, fs_ugi)
    path = check_hadoop_path(path, hadoop_bin, fs_name, fs_ugi)
    cmd = make_base_cmd(hadoop_bin, fs_name, fs_ugi)
    cmd += " -mkdir %s" % path
    cmd += " 2>%s" % ERR_LOG
    ret = os.system(cmd)
    return ret


def exists(path, hadoop_bin=None, fs_name=None, fs_ugi=None):
    """hadoop exists"""
    hadoop_bin, fs_name, fs_ugi = parse_account(hadoop_bin, fs_name, fs_ugi)
    path = check_hadoop_path(path, hadoop_bin, fs_name, fs_ugi)
    cmd = make_base_cmd(hadoop_bin, fs_name, fs_ugi)
    cmd += " -test -e " + path
    cmd += " 2>%s ; echo $?" % ERR_LOG
    ret = int(os.popen(cmd).read().strip())
    ret = True if ret == 0 else False
    return ret


def rm(path, hadoop_bin=None, fs_name=None, fs_ugi=None):
    """hadoop remove"""
    hadoop_bin, fs_name, fs_ugi = parse_account(hadoop_bin, fs_name, fs_ugi)
    path = check_hadoop_path(path, hadoop_bin, fs_name, fs_ugi)
    cmd = make_base_cmd(hadoop_bin, fs_name, fs_ugi)
    cmd += " -rmr %s" % path
    cmd += " 2>%s" % ERR_LOG

    if exists(path):
        ret = os.system(cmd)
        return ret
    else:
        return 0


def open(filename, hadoop_bin=None, fs_name=None, fs_ugi=None):
    """hadoop open file"""
    hadoop_bin, fs_name, fs_ugi = parse_account(hadoop_bin, fs_name, fs_ugi)
    filename = check_hadoop_path(filename, hadoop_bin, fs_name, fs_ugi)
    cmd = make_base_cmd(hadoop_bin, fs_name, fs_ugi)
    cmd += " -cat %s" % filename
    cmd += " 2>%s" % ERR_LOG
    p = os.popen(cmd)
    return p


def gz_open(filename, hadoop_bin=None, fs_name=None, fs_ugi=None):
    """hadoop open gz file"""
    hadoop_bin, fs_name, fs_ugi = parse_account(hadoop_bin, fs_name, fs_ugi)
    filename = check_hadoop_path(filename, hadoop_bin, fs_name, fs_ugi)
    cmd = make_base_cmd(hadoop_bin, fs_name, fs_ugi)
    cmd += " -text %s" % filename
    cmd += " 2>%s" % ERR_LOG
    p = os.popen(cmd)
    return p


def mv(src, dest, hadoop_bin=None, fs_name=None, fs_ugi=None):
    """hadoop move"""
    hadoop_bin, fs_name, fs_ugi = parse_account(hadoop_bin, fs_name, fs_ugi)
    src = check_hadoop_path(src, hadoop_bin, fs_name, fs_ugi)
    dest = check_hadoop_path(dest, hadoop_bin, fs_name, fs_ugi)
    if exists(dest):
        rm(dest)

    cmd = make_base_cmd(hadoop_bin, fs_name, fs_ugi)
    cmd += " -mv %s %s" % (src, dest)
    cmd += " 2>%s" % ERR_LOG
    ret = os.system(cmd)
    return ret


def get(src, dest, hadoop_bin=None, fs_name=None, fs_ugi=None):
    """ hadoop download file"""
    hadoop_bin, fs_name, fs_ugi = parse_account(hadoop_bin, fs_name, fs_ugi)
    src = check_hadoop_path(src, hadoop_bin, fs_name, fs_ugi)
    cmd = make_base_cmd(hadoop_bin, fs_name, fs_ugi)
    cmd += " -get %s %s" % (src, dest)
    cmd += " 2>%s" % ERR_LOG
    ret = os.system(cmd)
    return ret


def put(src, dest, hadoop_bin=None, fs_name=None, fs_ugi=None):
    """hadoop upload file"""
    hadoop_bin, fs_name, fs_ugi = parse_account(hadoop_bin, fs_name, fs_ugi)
    dest = check_hadoop_path(dest, hadoop_bin, fs_name, fs_ugi)
    cmd = make_base_cmd(hadoop_bin, fs_name, fs_ugi)
    cmd += " -put %s %s" % (src, dest)
    cmd += " 2>%s" % ERR_LOG
    ret = os.system(cmd)
    return ret


def replace(src, dest, hadoop_bin=None, fs_name=None, fs_ugi=None):
    """hadoop replace"""
    hadoop_bin, fs_name, fs_ugi = parse_account(hadoop_bin, fs_name, fs_ugi)
    src = check_hadoop_path(src, fs_name, fs_ugi)
    dest = check_hadoop_path(dest, fs_name, fs_ugi)

    tmp = dest + "_" + str(int(time.time()))
    cmd = make_base_cmd(hadoop_bin, fs_name, fs_ugi)
    cmd += " -mv " + dest + " " + tmp + " && "

    cmd += make_base_cmd(hadoop_bin, fs_name, fs_ugi)
    cmd += " -put " + src + " " + dest + " && "

    cmd += make_base_cmd(hadoop_bin, fs_name, fs_ugi)
    cmd += " -rmr " + tmp
    ret = os.system(cmd)
    return ret
