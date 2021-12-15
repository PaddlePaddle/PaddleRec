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
import shutil
from pathlib import Path
import warnings
import logging
import numpy as np
import math
import time
import datetime
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


def is_local(path):
    if path.startswith("hdfs:") or path.startswith("afs:"):
        return False
    else:
        return True


def file_ls(path_array, client):
    result = []
    for path in path_array:
        if is_local(path):
            cur_path = os.listdir(path)
        else:
            cur_path = client.ls_dir(path)[1]
        if len(cur_path) > 0:
            result += [os.path.join(path, i) for i in cur_path]
    logger.info("file ls result = {}".format(result))
    return result


def data_ready(data_donefile, sleep_second, client):
    while True:
        if is_local(data_donefile):
            donefile = Path(data_donefile)
            if donefile.is_file():
                return
        else:
            if client.is_file(data_donefile):
                return
        logger.info("wait for data ready: {}".format(data_donefile))
        time.sleep(sleep_second)


def get_next_day(day):
    tmp_date = datetime.datetime.strptime(day, '%Y%m%d').date()
    next_day = tmp_date + datetime.timedelta(days=1)
    next_day = next_day.strftime('%Y%m%d')
    return next_day


def get_online_pass_interval(split_interval, split_per_pass,
                             is_data_hourly_placed):
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


def load_model(model_path, mode, client):
    if not is_local(model_path) and (mode == 1 or mode == 2):
        local_path = "./dnn_plugin"
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        os.mkdir(local_path)
        client.download(model_path + "/dnn_plugin", local_path)
    fleet.load_model(model_path, mode)


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


def save_batch_model(exe, output_path, day):
    """
    save batch model

    Args:
        output_path(str): output path
        day(str|int): training day
    """
    day = str(day)
    suffix_name = "/%s/0/" % day
    model_path = output_path + suffix_name
    logger.info("going to save_batch_model %s" % model_path)
    fleet.save_persistables(exe, model_path, mode=3)


def write_model_donefile(output_path,
                         day,
                         pass_id,
                         xbox_base_key,
                         client,
                         donefile_name="donefile.txt"):
    """
    write donefile when save model

    Args:
        output_path(str): output path
        day(str|int): training day
        pass_id(str|int): training pass id
        xbox_base_key(str|int): xbox base key
        client(HDFSClient): hadoop client
        donefile_name(str): donefile name, default is "donefile.txt"r
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
        if not is_local(model_path):
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
                pre_content = f.read().strip("\n")
            logger.info("pre_content = {}".format(pre_content))
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


def get_last_save_model(output_path, client):
    """
    get last saved model info from donefile.txt

    Args:
        output_path(str): output path
        client(HDFSClient): hadoop client

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
    if not is_local(donefile_path):
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
            pre_content = f.read().strip("\n")
        content = pre_content.split("\n")[-1].split("\t")
        last_save_day = int(content[0])
        last_save_pass = int(content[3])
        last_path = content[2]
        xbox_base_key = int(content[1])
        return [last_save_day, last_save_pass, last_path, xbox_base_key]


def get_last_save_xbox_base(output_path, client):
    """
    get last saved base xbox info from xbox_base_done.txt

    Args:
        output_path(str): output path
        client(HDFSClient): hadoop client

    Returns:
        [last_save_day, last_path, xbox_base_key]
        last_save_day(int): day of saved model
        last_path(str): model path
        xbox_base_key(int): xbox key
    """

    donefile_path = output_path + "/xbox_base_done.txt"
    if not is_local(donefile_path):
        if not client.is_file(donefile_path):
            return [-1, -1, int(time.time())]
        pre_content = client.cat(donefile_path)
        last_dict = json.loads(pre_content.split("\n")[-1])
        last_day = int(last_dict["input"].split("/")[-3])
        last_path = "/".join(last_dict["input"].split("/")[:-1])
        xbox_base_key = int(last_dict["key"])
        return [last_day, last_path, xbox_base_key]
    else:
        file = Path(donefile_path)
        if not file.is_file():
            return [-1, -1, int(time.time())]
        with open(donefile_path, encoding='utf-8') as f:
            pre_content = f.read().strip("\n")
        last_line = pre_content.split("\n")[-1]
        last_dict = json.loads(last_line)
        last_day = int(last_dict["input"].split("/")[-3])
        last_path = "/".join(last_dict["input"].split("/")[:-1])
        xbox_base_key = int(last_dict["key"])
        return [last_day, last_path, xbox_base_key]


def get_last_save_xbox(output_path, client):
    """
    get last saved xbox info from xbox_patch_done.txt

    Args:
        output_path(str): output path
        client(HDFSClient): hadoop client

    Returns:
        [last_save_day, last_save_pass, last_path, xbox_base_key]
        last_save_day(int): day of saved model
        last_save_pass(int): pass id of saved
        last_path(str): model path
        xbox_base_key(int): xbox key

    """
    donefile_path = output_path + "/xbox_patch_done.txt"
    if not is_local(donefile_path):
        if not client.is_file(donefile_path):
            return [-1, -1, "", int(time.time())]
        logger.info("get_last_save_xbox donefile_path {} is file".format(
            donefile_path))
        pre_content = client.cat(donefile_path)
        logger.info("get_last_save_xbox get a pre_content = ", pre_content)
        last_dict = json.loads(pre_content.split("\n")[-1])
        last_day = int(last_dict["input"].split("/")[-3])
        last_pass = int(last_dict["input"].split("/")[-2].split("-")[-1])
        last_path = "/".join(last_dict["input"].split("/")[:-1])
        xbox_base_key = int(last_dict["key"])
        return [last_day, last_pass, last_path, xbox_base_key]
    else:
        file = Path(donefile_path)
        if not file.is_file():
            return [-1, -1, "", int(time.time())]
        with open(donefile_path, encoding='utf-8') as f:
            pre_content = f.read().strip("\n")
        last_line = pre_content.split("\n")[-1]
        last_dict = json.loads(last_line)
        last_day = int(last_dict["input"].split("/")[-3])
        last_pass = int(last_dict["input"].split("/")[-2].split("-")[-1])
        last_path = "/".join(last_dict["input"].split("/")[:-1])
        xbox_base_key = int(last_dict["key"])
        return [last_day, last_pass, last_path, xbox_base_key]


def save_xbox_model(output_path, day, pass_id, exe, feed_vars, target_vars,
                    client):
    """
    save xbox model

    Args:
        output_path(str): output path
        day(str|int): training day
        pass_id(str|int): training pass id
        exe(Executor)
        feed_vars(list): feed var list for inference model
        target_vars: target var list for inference model
        client(HDFSClient): hadoop client
    """
    if pass_id != -1:
        mode = 1
        suffix_name = "/%s/delta-%s/" % (day, pass_id)
        model_path = output_path.rstrip("/") + suffix_name
    else:
        mode = 2
        suffix_name = "/%s/base/" % day
        model_path = output_path.rstrip("/") + suffix_name
    fleet.save_inference_model(
        exe,
        model_path, [feed.name for feed in feed_vars],
        target_vars,
        mode=mode)
    if not is_local(model_path):
        client.upload("./dnn_plugin", model_path)


def write_xbox_donefile(output_path,
                        day,
                        pass_id,
                        xbox_base_key,
                        client,
                        hadoop_fs_name="",
                        monitor_data="",
                        donefile_name=None):
    """
    write xbox donefile when save xbox model

    Args:
        output_path(str): output path
        day(str|int): training day
        pass_id(str|int): training pass id
        xbox_base_key(str|int): xbox base key
        client(HDFSClient): hadoop client
        donefile_name(str): donefile name, default is "xbox_patch_done.txt"
    """
    day = str(day)
    pass_id = str(pass_id)
    xbox_base_key = int(xbox_base_key)
    mode = None

    if pass_id != "-1":
        mode = "patch"
        suffix_name = "/%s/delta-%s/" % (day, pass_id)
        model_path = output_path.rstrip("/") + suffix_name
        if donefile_name is None:
            donefile_name = "xbox_patch_done.txt"
    else:
        mode = "base"
        suffix_name = "/%s/base/" % day
        model_path = output_path.rstrip("/") + suffix_name
        if donefile_name is None:
            donefile_name = "xbox_base_done.txt"

    if fleet.worker_index() == 0:
        donefile_path = output_path + "/" + donefile_name
        xbox_str = _get_xbox_str(
            model_path=model_path,
            xbox_base_key=xbox_base_key,
            hadoop_fs_name=hadoop_fs_name,
            monitor_data=monitor_data,
            mode=mode)
        if not is_local(donefile_path):
            if client.is_file(donefile_path):
                pre_content = client.cat(donefile_path)
                last_line = pre_content.split("\n")[-1]
                if last_line == '':
                    last_line = pre_content.split("\n")[-2]
                last_dict = json.loads(last_line)
                last_day = last_dict["input"].split("/")[-3]
                last_pass = last_dict["input"].split("/")[-2].split("-")[-1]
                exist = False
                if int(day) < int(last_day) or \
                        int(day) == int(last_day) and \
                        int(pass_id) <= int(last_pass):
                    exist = True
                if not exist:
                    with open(donefile_name, "w") as f:
                        f.write(pre_content + "\n")
                        f.write(xbox_str + "\n")
                    client.delete(donefile_path)
                    client.upload(
                        donefile_name,
                        output_path,
                        multi_processes=1,
                        overwrite=False)
                    logger.info("write %s/%s %s success" % \
                                (day, pass_id, donefile_name))
                else:
                    logger.info("do not write %s because %s/%s already "
                                "exists" % (donefile_name, day, pass_id))
            else:
                with open(donefile_name, "w") as f:
                    f.write(xbox_str + "\n")
                client.upload(
                    donefile_name,
                    output_path,
                    multi_processes=1,
                    overwrite=False)
                logger.info("write %s/%s %s success" % \
                            (day, pass_id, donefile_name))
        else:
            file = Path(donefile_path)
            if not file.is_file():
                with open(donefile_path, "w") as f:
                    f.write(xbox_str + "\n")
                return
            with open(donefile_path, encoding='utf-8') as f:
                pre_content = f.read().strip("\n")
            exist = False
            last_line = pre_content.split("\n")[-1]
            last_dict = json.loads(last_line, strict=False)
            last_day = last_dict["input"].split("/")[-3]
            last_pass = last_dict["input"].split("/")[-2].split("-")[-1]
            if int(day) < int(last_day) or \
                    int(day) == int(last_day) and \
                    int(pass_id) <= int(last_pass):
                exist = True
            if not exist:
                with open(donefile_path, "w") as f:
                    f.write(pre_content + "\n")
                    f.write(xbox_str + "\n")


def _get_xbox_str(model_path,
                  xbox_base_key,
                  hadoop_fs_name="",
                  monitor_data="",
                  mode="patch"):
    xbox_dict = collections.OrderedDict()
    if mode == "base":
        xbox_dict["id"] = str(xbox_base_key)
    elif mode == "patch":
        xbox_dict["id"] = str(int(time.time()))
    else:
        logger.info("warning: unknown mode %s, set it to patch" % mode)
        mode = "patch"
        xbox_dict["id"] = str(int(time.time()))
    xbox_dict["key"] = str(xbox_base_key)
    if is_local(model_path):
        hadoop_fs_name = ""
    else:
        model_path = model_path[model_path.find(":") + 1:]
    xbox_dict["input"] = hadoop_fs_name + model_path.rstrip("/") + "/001"
    xbox_dict["monitor_data"] = monitor_data
    return json.dumps(xbox_dict)
