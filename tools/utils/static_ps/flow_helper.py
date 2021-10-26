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

from __future__ import print_function
import os
import sys
from pathlib import Path
import warnings
import logging
import numpy as np
import math
import time
import json
import collections
import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
import common

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def file_ls(path_array, train_local, client):
    result = []
    if train_local:
        for path in path_array:
            for root, ds, fs in os.walk(path):
                for f in fs:
                    fullname = os.path.join(root, f)
                    result.append(fullname)
    else:
        for i in path_array:
            cur_path = client.ls_dir(i)[1]
            if len(cur_path) > 0:
                i = i.strip("/")
                result += [i.rstrip("/") + "/" + j for j in cur_path]
    logger.info("file ls result = {}".format(result))
    return result


def data_ready(train_local, client, data_donefile, sleep_second):
    while True:
        if train_local:
            donefile = Path(data_donefile)
            if donefile.is_file():
                return
        else:
            if client.is_file(data_donefile):
                return
        logger.info("wait for data ready: {}".format(data_donefile))
        time.sleep(sleep_second)


def get_next_day(day):
    return os.popen('date -d"%s' % day + ' +1 days" +"%Y%m%d"').read().strip()


def get_online_pass_interval(start_day, end_day, split_interval,
                             split_per_pass, is_data_hourly_placed):
    # days = os.popen("echo -n " + days).read().split(" ")
    # hours = os.popen("echo -n " + hours).read().split(" ")
    split_interval = int(split_interval)
    split_per_pass = int(split_per_pass)
    splits_per_day = 24 * 60 // split_interval
    pass_per_day = splits_per_day // split_per_pass
    left_train_hour = 0
    right_train_hour = 23

    start = 0
    split_path = []
    for i in range(splits_per_day):
        h = start // 60
        m = start % 60
        if h < left_train_hour or h > right_train_hour:
            start += split_interval
            continue
        if is_data_hourly_placed:
            split_path.append("%02d" % h)
        else:
            split_path.append("%02d%02d" % (h, m))
        start += split_interval

    start = 0
    online_pass_interval = []
    for i in range(pass_per_day):
        online_pass_interval.append([])
        for j in range(start, start + split_per_pass):
            online_pass_interval[i].append(split_path[j])
        start += split_per_pass

    return online_pass_interval


def save_model(exe, output_path, day, pass_id, mode=0):
    """
    Args:
        output_path(str): output path
        day(str|int): training day
        pass_id(str|int): training pass id

    """
    day = str(day)
    pass_id = str(pass_id)
    suffix_name = "/%s/%s/" % (day, pass_id)
    model_path = output_path + suffix_name
    logger.info("going to save_model %s" % model_path)
    fleet.save_persistables(exe, model_path, None, mode=mode)


def save_inference_model(exe,
                         output_path,
                         day,
                         pass_id,
                         feed_vars,
                         target_vars,
                         mode=0):
    """
    Args:
        output_path(str): output path
        day(str|int): training day
        pass_id(str|int): training pass id

    """
    day = str(day)
    pass_id = str(pass_id)
    suffix_name = "/%s/inference_model_%s/" % (day, pass_id)
    model_path = output_path + suffix_name
    logger.info("going to save_inference_model %s" % model_path)
    fleet.save_inference_model(
        exe,
        model_path, [feed.name for feed in feed_vars],
        target_vars,
        mode=0)


def save_batch_model(exe, output_path, day):
    """
    save batch model

    Args:
        output_path(str): output path
        day(str|int): training day

    Examples:
        .. code-block:: python

        from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
        fleet_util = FleetUtil()
        fleet_util.save_batch_model("hdfs:/my/path", 20190722)

    """
    day = str(day)
    suffix_name = "/%s/0/" % day
    model_path = output_path + suffix_name
    fleet.save_persistables(exe, model_path, mode=3)


def write_model_donefile(output_path,
                         day,
                         pass_id,
                         xbox_base_key,
                         train_local,
                         client,
                         donefile_name="donefile.txt"):
    """
    write donefile when save model

    Args:
        output_path(str): output path
        day(str|int): training day
        pass_id(str|int): training pass id
        xbox_base_key(str|int): xbox base key
        hadoop_fs_name(str): hdfs/afs fs name
        hadoop_fs_ugi(str): hdfs/afs fs ugi
        hadoop_home(str): hadoop home, default is "$HADOOP_HOME"
        donefile_name(str): donefile name, default is "donefile.txt"


    """
    day = str(day)
    pass_id = str(pass_id)
    xbox_base_key = int(xbox_base_key)

    if pass_id != "-1":
        suffix_name = "/%s/%s/" % (day, pass_id)
        model_path = output_path.rstrip("/") + suffix_name
    else:
        suffix_name = "/%s/0/" % day
        model_path = output_path.rstrip("/") + suffix_name

    if fleet.worker_index() == 0:
        donefile_path = output_path + "/" + donefile_name
        content = "%s\t%lu\t%s\t%s\t%d" % (day, xbox_base_key, \
                                            model_path, pass_id, 0)
        if not train_local:
            if client.is_file(donefile_path):
                pre_content = client.cat(donefile_path)
                pre_content_list = pre_content.split("\n")
                day_list = [i.split("\t")[0] for i in pre_content_list]
                pass_list = [i.split("\t")[3] for i in pre_content_list]
                exist = False
                for i in range(len(day_list)):
                    if int(day) == int(day_list[i]) and \
                            int(pass_id) == int(pass_list[i]):
                        exist = True
                        break
                if not exist:
                    with open(donefile_name, "w") as f:
                        f.write(pre_content + "\n")
                        f.write(content + "\n")
                    client.delete(donefile_path)
                    client.upload(donefile_name, output_path)
                    logger.info("write %s/%s %s succeed" % \
                                (day, pass_id, donefile_name))
                else:
                    logger.info("not write %s because %s/%s already "
                                "exists" % (donefile_name, day, pass_id))
            else:
                with open(donefile_name, "w") as f:
                    f.write(content + "\n")
                client.upload(donefile_name, output_path)
                logger.info("write %s/%s %s succeed" % \
                            (day, pass_id, donefile_name))
        else:
            file = Path(donefile_path)
            logger.info("model done file path = {}, content = {}".format(
                donefile_path, content))
            if not file.is_file():
                logger.info(" {} doesn't exist ".format(donefile_path))
                with open(donefile_path, "w") as f:
                    f.write(content + "\n")
                return
            with open(donefile_path, encoding='utf-8') as f:
                pre_content = f.read()
            logger.info("pre_content = {}".format(pre_content))
            lines = pre_content.split("\n")
            day_list = []
            pass_list = []
            for i in lines:
                if i == "":
                    continue
                arr = i.split("\t")
                day_list.append(arr[0])
                pass_list.append(arr[3])
            exist = False
            for i in range(len(day_list)):
                if int(day) == int(day_list[i]) and \
                        int(pass_id) == int(pass_list[i]):
                    exist = True
                    break
            if not exist:
                with open(donefile_path, "w") as f:
                    f.write(pre_content + "\n")
                    logger.info("write donefile {}".format(pre_content))
                    f.write(content + "\n")
                    logger.info("write donefile {}".format(content))
                logger.info("write %s/%s %s succeed" % \
                            (day, pass_id, donefile_name))
            else:
                logger.info("not write %s because %s/%s already "
                            "exists" % (donefile_name, day, pass_id))


def get_last_save_model(output_path, train_local, client):
    r"""
    get last saved model info from donefile.txt

    Args:
        output_path(str): output path
        hadoop_fs_name(str): hdfs/afs fs_name
        hadoop_fs_ugi(str): hdfs/afs fs_ugi
        hadoop_home(str): hadoop home, default is "$HADOOP_HOME"

    Returns:
        [last_save_day, last_save_pass, last_path, xbox_base_key]
        last_save_day(int): day of saved model
        last_save_pass(int): pass id of saved
        last_path(str): model path
        xbox_base_key(int): xbox key

    """
    last_save_day = -1
    last_save_pass = -1
    last_path = ""
    donefile_path = output_path + "/donefile.txt"
    if not train_local:
        if not client.is_file(donefile_path):
            return [-1, -1, "", int(time.time())]
        content = client.cat(donefile_path)
        content = content.split("\n")[-1].split("\t")
        last_save_day = int(content[0])
        last_save_pass = int(content[3])
        last_path = content[2]
        xbox_base_key = int(content[1])
        return [last_save_day, last_save_pass, last_path, xbox_base_key]
    else:
        file = Path(donefile_path)
        if not file.is_file():
            return [-1, -1, "", int(time.time())]
        with open(donefile_path, encoding='utf-8') as f:
            pre_content = f.read()
        exist = False
        last_line = pre_content.split("\n")[-1]
        if last_line == '':
            last_line = pre_content.split("\n")[-2]
        content = last_line.split("\n")[-1].split("\t")
        last_save_day = int(content[0])
        last_save_pass = int(content[3])
        last_path = content[2]
        xbox_base_key = int(content[1])
        return [last_save_day, last_save_pass, last_path, xbox_base_key]
