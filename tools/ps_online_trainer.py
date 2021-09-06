# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import math
import random
import numpy as np
from pathlib import Path
from utils.static_ps.reader_helper import get_reader, get_example_num, get_file_list, get_word_num
from utils.static_ps.program_helper import get_model, get_strategy, set_dump_config
from utils.static_ps.common import YamlHelper, is_distributed_env
import argparse
import time
import sys
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle
import collections
import copy
import json
import logging
import os
import sys
import time
import common
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.log_helper import get_logger
from paddle.distributed.fleet.utils.fs import LocalFS, HDFSClient

OpRole = core.op_proto_and_checker_maker.OpRole
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Training(object):
    def __init__(self, config):
        self.metrics = {}
        self.config = config
        self.exe = None
        self.reader_type = "QueueDataset"
        self.split_interval = config.get("runner.split_interval")
        self.split_per_pass = config.get("runner.split_per_pass")
        self.save_delta_frequency = config.get("runner.save_delta_frequency",
                                               6)
        self.checkpoint_per_pass = config.get("runner.checkpoint_per_pass", 6)
        self.save_first_base = config.get("runner.save_first_base", False)
        self.start_day = config.get("runner.start_day")
        self.end_day = config.get("runner.end_day")
        self.save_model_path = self.config.get("runner.model_save_path")
        self.data_path = self.config.get("runner.train_data_dir")
        if config.get("runner.fs_client.uri") is not None:
            self.hadoop_config = {}
            for key in ["uri", "user", "passwd", "hadoop_bin"]:
                self.hadoop_config[key] = config.get("runner.fs_client." + key,
                                                     "")
            self.hadoop_fs_name = self.hadoop_config.get("uri")
            self.hadoop_fs_ugi = self.hadoop_config.get(
                "user") + "," + self.hadoop_config.get("passwd")
            # prefix = "hdfs:/" if self.hadoop_fs_name.startswith("hdfs:") else "afs:/"
            # self.save_model_path = prefix + self.save_model_path.strip("/")
        else:
            self.hadoop_fs_name, self.hadoop_fs_ugi = None, None
        self.train_local = self.hadoop_fs_name is None or self.hadoop_fs_ugi is None

    def run(self):
        os.environ["PADDLE_WITH_GLOO"] = "1"
        role = role_maker.PaddleCloudRoleMaker(init_gloo=True)
        fleet.init(role)
        self.init_network()
        if fleet.is_server():
            self.run_server()
        elif fleet.is_worker():
            self.run_worker()
            fleet.stop_worker()
        logger.info("successfully completed running, Exit.")

    def get_next_day(self, day):
        return os.popen('date -d"%s' % day + ' +1 days" +"%Y%m%d"').read(
        ).strip()

    def save_model(self, output_path, day, pass_id, mode=0):
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
        fleet.save_persistables(self.exe, model_path, None, mode=mode)

    def write_model_donefile(self,
                             output_path,
                             day,
                             pass_id,
                             xbox_base_key,
                             hadoop_fs_name,
                             hadoop_fs_ugi,
                             hadoop_home="$HADOOP_HOME",
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
            if not self.train_local:
                configs = {
                    "fs.default.name": hadoop_fs_name,
                    "hadoop.job.ugi": hadoop_fs_ugi
                }
                client = HDFSClient(hadoop_home, configs)
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

    def get_last_save_xbox_base(self,
                                output_path,
                                hadoop_fs_name,
                                hadoop_fs_ugi,
                                hadoop_home="$HADOOP_HOME"):
        r"""
        get last saved base xbox info from xbox_base_done.txt

        Args:
            output_path(str): output path
            hadoop_fs_name(str): hdfs/afs fs_name
            hadoop_fs_ugi(str): hdfs/afs fs_ugi
            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"

        Returns:
            [last_save_day, last_path, xbox_base_key]
            last_save_day(int): day of saved model
            last_path(str): model path
            xbox_base_key(int): xbox key

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              last_save_day, last_path, xbox_base_key = \
                  fleet_util.get_last_save_xbox_base("hdfs:/my/path", 20190722,
                                                     88)

        """

        donefile_path = output_path + "/xbox_base_done.txt"
        if not self.train_local:
            configs = {
                "fs.default.name": hadoop_fs_name,
                "hadoop.job.ugi": hadoop_fs_ugi
            }
            client = HDFSClient(hadoop_home, configs)
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
                pre_content = f.read()
            last_line = pre_content.split("\n")[-1]
            if last_line == '':
                last_line = pre_content.split("\n")[-2]
            last_dict = json.loads(last_line)
            last_day = int(last_dict["input"].split("/")[-3])
            last_path = "/".join(last_dict["input"].split("/")[:-1])
            xbox_base_key = int(last_dict["key"])
            return [last_day, last_path, xbox_base_key]

    def clear_metrics(self, scope, var_list, var_types):
        from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
        fleet_util = FleetUtil()
        for i in range(len(var_list)):
            fleet_util.set_zero(
                var_list[i].name, scope, param_type=var_types[i])

    def get_global_auc(self,
                       scope=fluid.global_scope(),
                       stat_pos="_generated_var_2",
                       stat_neg="_generated_var_3"):
        """
        Get global auc of all distributed workers.

        Args:
            scope(Scope): Scope object, default is fluid.global_scope()
            stat_pos(str): name of auc pos bucket Variable
            stat_neg(str): name of auc neg bucket Variable

        Returns:
            auc_value(float), total_ins_num(int)

        """
        if scope.find_var(stat_pos) is None or scope.find_var(
                stat_neg) is None:
            self.rank0_print("not found auc bucket")
            return None
        fleet.barrier_worker()
        # auc pos bucket
        pos = np.array(scope.find_var(stat_pos).get_tensor())
        # auc pos bucket shape
        old_pos_shape = np.array(pos.shape)
        # reshape to one dim
        pos = pos.reshape(-1)
        #global_pos = np.copy(pos) * 0
        # mpi allreduce
        global_pos = fleet.util.all_reduce(pos)
        # reshape to its original shape
        global_pos = global_pos.reshape(old_pos_shape)
        print('debug global auc global_pos: ', global_pos)

        # auc neg bucket
        neg = np.array(scope.find_var(stat_neg).get_tensor())
        old_neg_shape = np.array(neg.shape)
        neg = neg.reshape(-1)
        #global_neg = np.copy(neg) * 0
        global_neg = fleet.util.all_reduce(neg)
        global_neg = global_neg.reshape(old_neg_shape)
        print('debug global auc global_neg: ', global_neg)

        # calculate auc
        num_bucket = len(global_pos[0])
        area = 0.0
        pos = 0.0
        neg = 0.0
        new_pos = 0.0
        new_neg = 0.0
        total_ins_num = 0
        for i in range(num_bucket):
            index = num_bucket - 1 - i
            new_pos = pos + global_pos[0][index]
            total_ins_num += global_pos[0][index]
            new_neg = neg + global_neg[0][index]
            total_ins_num += global_neg[0][index]
            area += (new_neg - neg) * (pos + new_pos) / 2
            pos = new_pos
            neg = new_neg

        if pos * neg == 0 or total_ins_num == 0:
            auc_value = 0.5
        else:
            auc_value = area / (pos * neg)

        fleet.barrier_worker()
        return auc_value

    def save_xbox_model(self,
                        output_path,
                        day,
                        pass_id,
                        hadoop_fs_name,
                        hadoop_fs_ugi,
                        hadoop_home="$HADOOP_HOME"):
        if pass_id != -1:
            mode = 1
            suffix_name = "/%s/delta-%s/" % (day, pass_id)
            model_path = output_path.rstrip("/") + suffix_name
        else:
            mode = 2
            suffix_name = "/%s/base/" % day
            model_path = output_path.rstrip("/") + suffix_name
        infer_program_path = model_path + "dnn_plugin/"
        if self.train_local:
            fleet.save_inference_model(
                self.exe, infer_program_path,
                [feed.name for feed in self.inference_model_feed_vars],
                self.predict)  # 0 save checkpoints
        else:
            model_name = "inference_model"
            fleet.save_inference_model(
                self.exe, model_name,
                [feed.name for feed in self.inference_model_feed_vars],
                self.predict)
            configs = {
                "fs.default.name": hadoop_fs_name,
                "hadoop.job.ugi": hadoop_fs_ugi
            }
            client = HDFSClient(hadoop_home, configs)

            if pass_id == "-1":
                dest = "%s/%s/base/dnn_plugin/" % (output_path, day)
            else:
                dest = "%s/%s/delta-%s/dnn_plugin/" % (output_path, day,
                                                       pass_id)
            if not client.is_exist(dest):
                client.mkdirs(dest)

            client.upload(model_name, dest, multi_processes=5, overwrite=True)
        fleet.save_persistables(
            executor=self.exe, dirname=model_path, mode=mode)
        logger.info("save_persistables in %s" % model_path)

    def get_last_save_model(self,
                            output_path,
                            hadoop_fs_name,
                            hadoop_fs_ugi,
                            hadoop_home="$HADOOP_HOME"):
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
        if not self.train_local:
            configs = {
                "fs.default.name": hadoop_fs_name,
                "hadoop.job.ugi": hadoop_fs_ugi
            }
            client = HDFSClient(hadoop_home, configs)
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

    def get_last_save_xbox(self,
                           output_path,
                           hadoop_fs_name,
                           hadoop_fs_ugi,
                           hadoop_home="$HADOOP_HOME"):
        r"""
        get last saved xbox info from xbox_patch_done.txt

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
        donefile_path = output_path + "/xbox_patch_done.txt"
        if not self.train_local:
            configs = {
                "fs.default.name": hadoop_fs_name,
                "hadoop.job.ugi": hadoop_fs_ugi
            }
            client = HDFSClient(hadoop_home, configs)
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
                pre_content = f.read()
            last_line = pre_content.split("\n")[-1]
            if last_line == '':
                last_line = pre_content.split("\n")[-2]
            last_dict = json.loads(last_line)
            last_day = int(last_dict["input"].split("/")[-3])
            last_pass = int(last_dict["input"].split("/")[-2].split("-")[-1])
            last_path = "/".join(last_dict["input"].split("/")[:-1])
            xbox_base_key = int(last_dict["key"])
            return [last_day, last_pass, last_path, xbox_base_key]

    def get_online_pass_interval(self, start_day, end_day, split_interval,
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

    def write_xbox_donefile(self,
                            output_path,
                            day,
                            pass_id,
                            model_base_key,
                            hadoop_fs_name,
                            hadoop_fs_ugi,
                            hadoop_home="$HADOOP_HOME",
                            donefile_name=None):
        day = str(day)
        pass_id = str(pass_id)
        xbox_base_key = int(model_base_key)
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
            xbox_str = self._get_xbox_str(
                model_path=model_path, xbox_base_key=xbox_base_key, mode=mode)
            if not self.train_local:
                configs = {
                    "fs.default.name": hadoop_fs_name,
                    "hadoop.job.ugi": hadoop_fs_ugi
                }
                client = HDFSClient(hadoop_home, configs)
                if client.is_file(donefile_path):
                    pre_content = client.cat(donefile_path)
                    last_line = pre_content.split("\n")[-1]
                    if last_line == '':
                        last_line = pre_content.split("\n")[-2]
                    last_dict = json.loads(last_line)
                    last_day = last_dict["input"].split("/")[-3]
                    last_pass = last_dict["input"].split("/")[-2].split("-")[
                        -1]
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
                    pre_content = f.read()
                exist = False
                last_line = pre_content.split("\n")[-1]
                if last_line == '':
                    last_line = pre_content.split("\n")[-2]
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

    def _get_xbox_str(self,
                      model_path,
                      xbox_base_key,
                      hadoop_fs_name=None,
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
        if model_path.startswith("hdfs:") or model_path.startswith("afs:"):
            model_path = model_path[model_path.find(":") + 1:]
        xbox_dict["input"] = ("" if hadoop_fs_name is None else hadoop_fs_name
                              ) + model_path.rstrip("/") + "/000"
        return json.dumps(xbox_dict)

    def get_global_metrics(self,
                           scope=fluid.global_scope(),
                           stat_pos_name="_generated_var_2",
                           stat_neg_name="_generated_var_3",
                           sqrerr_name="sqrerr",
                           abserr_name="abserr",
                           prob_name="prob",
                           q_name="q",
                           pos_ins_num_name="pos",
                           total_ins_num_name="total"):
        from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
        fleet_util = FleetUtil()
        if scope.find_var(stat_pos_name) is None or \
                scope.find_var(stat_neg_name) is None:
            fleet_util.rank0_print("not found auc bucket")
            return [None] * 9
        elif scope.find_var(sqrerr_name) is None:
            fleet_util.rank0_print("not found sqrerr_name=%s" % sqrerr_name)
            return [None] * 9
        elif scope.find_var(abserr_name) is None:
            fleet_util.rank0_print("not found abserr_name=%s" % abserr_name)
            return [None] * 9
        elif scope.find_var(prob_name) is None:
            fleet_util.rank0_print("not found prob_name=%s" % prob_name)
            return [None] * 9
        elif scope.find_var(q_name) is None:
            fleet_util.rank0_print("not found q_name=%s" % q_name)
            return [None] * 9
        elif scope.find_var(pos_ins_num_name) is None:
            fleet_util.rank0_print("not found pos_ins_num_name=%s" %
                                   pos_ins_num_name)
            return [None] * 9
        elif scope.find_var(total_ins_num_name) is None:
            fleet_util.rank0_print("not found total_ins_num_name=%s" % \
                                   total_ins_num_name)
            return [None] * 9

        # barrier worker to ensure all workers finished training
        fleet.barrier_worker()

        # get auc
        auc = self.get_global_auc(scope, stat_pos_name, stat_neg_name)
        pos = np.array(scope.find_var(stat_pos_name).get_tensor())
        # auc pos bucket shape
        old_pos_shape = np.array(pos.shape)
        # reshape to one dim
        pos = pos.reshape(-1)
        global_pos = np.copy(pos) * 0
        # mpi allreduce
        # fleet._role_maker._all_reduce(pos, global_pos)
        global_pos = fleet.util.all_reduce(pos)
        # reshape to its original shape
        global_pos = global_pos.reshape(old_pos_shape)
        # auc neg bucket
        neg = np.array(scope.find_var(stat_neg_name).get_tensor())
        old_neg_shape = np.array(neg.shape)
        neg = neg.reshape(-1)
        global_neg = np.copy(neg) * 0
        # fleet._role_maker._all_reduce(neg, global_neg)
        global_neg = fleet.util.all_reduce(neg)
        global_neg = global_neg.reshape(old_neg_shape)

        num_bucket = len(global_pos[0])

        def get_metric(name):
            metric = np.array(scope.find_var(name).get_tensor())
            old_metric_shape = np.array(metric.shape)
            metric = metric.reshape(-1)
            print(name, 'ori value:', metric)
            global_metric = np.copy(metric) * 0
            # fleet._role_maker._all_reduce(metric, global_metric)
            global_metric = fleet.util.all_reduce(metric)
            global_metric = global_metric.reshape(old_metric_shape)
            print(name, global_metric)
            return global_metric[0]

        global_sqrerr = get_metric(sqrerr_name)
        global_abserr = get_metric(abserr_name)
        global_prob = get_metric(prob_name)
        global_q_value = get_metric(q_name)
        # note: get ins_num from auc bucket is not actual value,
        # so get it from metric op
        pos_ins_num = get_metric(pos_ins_num_name)
        total_ins_num = get_metric(total_ins_num_name)
        neg_ins_num = total_ins_num - pos_ins_num

        mae = global_abserr / total_ins_num
        rmse = math.sqrt(global_sqrerr / total_ins_num)
        return_actual_ctr = pos_ins_num / total_ins_num
        predicted_ctr = global_prob / total_ins_num
        mean_predict_qvalue = global_q_value / total_ins_num
        copc = 0.0
        if abs(predicted_ctr > 1e-6):
            copc = return_actual_ctr / predicted_ctr

        # calculate bucket error
        last_ctr = -1.0
        impression_sum = 0.0
        ctr_sum = 0.0
        click_sum = 0.0
        error_sum = 0.0
        error_count = 0.0
        click = 0.0
        show = 0.0
        ctr = 0.0
        adjust_ctr = 0.0
        relative_error = 0.0
        actual_ctr = 0.0
        relative_ctr_error = 0.0
        k_max_span = 0.01
        k_relative_error_bound = 0.05
        for i in range(num_bucket):
            click = global_pos[0][i]
            show = global_pos[0][i] + global_neg[0][i]
            ctr = float(i) / num_bucket
            if abs(ctr - last_ctr) > k_max_span:
                last_ctr = ctr
                impression_sum = 0.0
                ctr_sum = 0.0
                click_sum = 0.0
            impression_sum += show
            ctr_sum += ctr * show
            click_sum += click
            if impression_sum == 0:
                continue
            adjust_ctr = ctr_sum / impression_sum
            if adjust_ctr == 0:
                continue
            relative_error = \
                math.sqrt((1 - adjust_ctr) / (adjust_ctr * impression_sum))
            if relative_error < k_relative_error_bound:
                actual_ctr = click_sum / impression_sum
                relative_ctr_error = abs(actual_ctr / adjust_ctr - 1)
                error_sum += relative_ctr_error * impression_sum
                error_count += impression_sum
                last_ctr = -1

        bucket_error = error_sum / error_count if error_count > 0 else 0.0

        return [
            auc, bucket_error, mae, rmse, return_actual_ctr, predicted_ctr,
            copc, mean_predict_qvalue, int(total_ins_num)
        ]

    def get_global_metrics_str(self, scope, metric_list, prefix):
        if len(metric_list) != 10:
            raise ValueError("len(metric_list) != 10, %s" % len(metric_list))

        auc, bucket_error, mae, rmse, actual_ctr, predicted_ctr, copc, \
        mean_predict_qvalue, total_ins_num = self.get_global_metrics( \
            scope, metric_list[2].name, metric_list[3].name, metric_list[4].name, metric_list[5].name, \
            metric_list[6].name, metric_list[7].name, metric_list[8].name, metric_list[9].name)
        metrics_str = "%s global AUC=%.6f BUCKET_ERROR=%.6f MAE=%.6f " \
                      "RMSE=%.6f Actural_CTR=%.6f Predicted_CTR=%.6f " \
                      "COPC=%.6f MEAN Q_VALUE=%.6f Ins number=%s" % \
                      (prefix, auc, bucket_error, mae, rmse, \
                       actual_ctr, predicted_ctr, copc, mean_predict_qvalue, \
                       total_ins_num)
        return metrics_str

    def init_network(self):
        self.model = get_model(self.config)
        self.input_data = self.model.create_feeds()
        self.inference_feed_var = self.model.create_feeds(is_infer=True)
        self.metrics = self.model.net(self.input_data)
        self.inference_target_var = self.model.inference_target_var
        self.predict = self.model.predict
        self.inference_model_feed_vars = self.model.inference_model_feed_vars
        logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))
        thread_stat_var_names = [
            self.model.auc_stat_list[2].name, self.model.auc_stat_list[3].name
        ]

        thread_stat_var_names += [i.name for i in self.model.metric_list]

        thread_stat_var_names = list(set(thread_stat_var_names))

        self.config['stat_var_names'] = thread_stat_var_names
        self.model.create_optimizer(get_strategy(self.config))

    def run_server(self):
        logger.info("Run Server Begin")
        # fleet.init_server(config.get("runner.warmup_model_path", "./warmup"))
        fleet.init_server(fs_client=self.hadoop_config)
        fleet.run_server()

    def file_ls(self, path_array):
        result = []
        if self.train_local:
            for path in path_array:
                for root, ds, fs in os.walk(path):
                    for f in fs:
                        fullname = os.path.join(root, f)
                        result.append(fullname)
        else:
            configs = {
                "fs.default.name": self.hadoop_fs_name,
                "hadoop.job.ugi": self.hadoop_fs_ugi
            }
            data_path = self.data_path
            hdfs_client = HDFSClient("$HADOOP_HOME", configs)
            for i in path_array:
                cur_path = hdfs_client.ls_dir(i)[1]
                #prefix = "hdfs:" if self.hadoop_fs_name.startswith("hdfs:") else "afs:"
                if len(cur_path) > 0:
                    i = i.strip("/")
                    #result += [prefix + i.rstrip("/") + "/" + j for j in cur_path]
                    result += [i.rstrip("/") + "/" + j for j in cur_path]
                    # #result += cur_path
                    # result += [data_path.rstrip("/") + "/" + j for j in cur_path]
        logger.info("file ls result = {}".format(result))
        return result

    def save_batch_model(self, output_path, day):
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
        fleet.save_persistables(None, model_path, mode=3)

    def prepare_dataset(self, day, pass_index):
        # dataset, file_list = get_reader(self.input_data, config)

        dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        # dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
        dataset.set_use_var(self.input_data)
        dataset.set_batch_size(self.config.get('runner.train_batch_size'))
        dataset.set_thread(self.config.get('runner.train_thread_num', 1))
        data_path = self.config.get("runner.train_data_dir")
        if not self.train_local:
            dataset.set_hdfs_config(self.hadoop_fs_name, self.hadoop_fs_ugi)
            logger.info("set hadoop_fs_name = {}, fs_ugi={}".format(
                self.hadoop_fs_name, self.hadoop_fs_ugi))
        cur_path = []
        for i in self.online_intervals[pass_index - 1]:
            cur_path.append(data_path.rstrip("/") + "/" + day + "/" + i)
        global_file_list = self.file_ls(cur_path)
        my_file_list = fleet.util.get_file_shard(global_file_list)
        logger.info("my_file_list = {}".format(my_file_list))
        dataset.set_filelist(my_file_list)
        pipe_command = self.config.get("runner.pipe_command")
        # dataset.set_pipe_command(self.config.get("runner.pipe_command"))
        utils_path = common.get_utils_file_path()
        dataset.set_pipe_command("{} {} {}".format(
            pipe_command, config.get("yaml_path"), utils_path))
        logger.info("pipe_command = " + "{} {} {}".format(
            pipe_command, config.get("yaml_path"), utils_path))
        dataset.load_into_memory()
        return dataset

    def run_worker(self):
        logger.info("Run Online Worker Begin")
        use_cuda = int(config.get("runner.use_gpu"))
        place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
        self.exe = paddle.static.Executor(place)
        self.exe.run(paddle.static.default_startup_program())
        fleet.init_worker()
        self.online_intervals = self.get_online_pass_interval(
            self.start_day, self.end_day, self.split_interval,
            self.split_per_pass, False)
        if self.train_local and self.save_model_path and (
                not os.path.exists(self.save_model_path)):
            os.makedirs(self.save_model_path)

        last_day, last_pass, last_path, model_base_key = self.get_last_save_model(
            self.save_model_path, self.hadoop_fs_name, self.hadoop_fs_ugi)
        logger.info(
            "get_last_save_model last_day = {}, last_pass = {}, last_path = {}, model_base_key = {}".
            format(last_day, last_pass, last_path, model_base_key))
        if last_day != -1:
            fleet.load_model(last_path, mode="0")

        save_first_base = self.save_first_base
        day = self.start_day

        # set_dump_config(paddle.static.default_main_program(), {
        #         "dump_fields_path": "./dump",
        #         "dump_fields": ["embedding_343.tmp_0"],
        #         "dump_param": []
        #     })
        # print("&&&&&&&&",paddle.static.default_main_program()._fleet_opt)
        while int(day) <= int(self.end_day):
            logger.info("training a new day {}, end_day = {}".format(
                day, self.end_day))
            if last_day != -1 and int(day) < last_day:
                day = int(self.get_next_day(day))
                continue
            # base_model_saved = False
            save_model_path = self.save_model_path
            for pass_id in range(len(self.online_intervals)):
                index = pass_id + 1
                if (last_day != -1 and int(day) == last_day) and (
                        last_pass != -1 and int(index) < last_pass):
                    # base_model_saved = True
                    continue
                if fleet.is_first_worker() and save_first_base:
                    save_first_base = False
                    last_base_day, last_base_path, tmp_xbox_base_key = \
                        self.get_last_save_xbox_base(save_model_path, self.hadoop_fs_name, self.hadoop_fs_ugi)
                    logger.info(
                        "get_last_save_xbox_base, last_base_day = {}, last_base_path = {}, tmp_xbox_base_key = {}".
                        format(last_base_day, last_base_path,
                               tmp_xbox_base_key))
                    if int(day) > last_base_day:
                        model_base_key = int(time.time())
                        self.save_xbox_model(save_model_path, day, -1,
                                             self.hadoop_fs_name,
                                             self.hadoop_fs_ugi)
                        self.write_xbox_donefile(
                            output_path=save_model_path,
                            day=day,
                            pass_id=-1,
                            model_base_key=model_base_key,
                            hadoop_fs_name=self.hadoop_fs_name,
                            hadoop_fs_ugi=self.hadoop_fs_ugi)
                    elif int(day) == last_base_day:
                        model_base_key = tmp_xbox_base_key

                logger.info("training a new day = {} new pass = {}".format(
                    day, pass_id))
                prepare_data_start_time = time.time()
                dataset = self.prepare_dataset(day, index)
                prepare_data_end_time = time.time()
                logger.info(
                    "Prepare Dataset Done, using time {} second.".format(
                        prepare_data_end_time - prepare_data_start_time))

                set_dump_config(paddle.static.default_main_program(), {
                    "dump_fields_path": './test_dump',
                    "dump_fields": [
                        'sparse_embedding_0.tmp_0@GRAD',
                        'sequence_pool_0.tmp_0@GRAD', 'concat_0.tmp_0@GRAD',
                        'concat_0.tmp_0', 'linear_6.tmp_1', 'relu_0.tmp_0',
                        'linear_7.tmp_1', 'relu_1.tmp_0', 'linear_8.tmp_1',
                        'relu_2.tmp_0', 'linear_9.tmp_1', 'relu_3.tmp_0',
                        'linear_10.tmp_1', 'relu_4.tmp_0', 'linear_11.tmp_1',
                        'sigmoid_0.tmp_0', 'clip_0.tmp_0',
                        'sigmoid_0.tmp_0@GRAD', 'clip_0.tmp_0@GRAD',
                        'linear_11.tmp_1@GRAD', 'linear_9.tmp_1@GRAD',
                        'linear_6.tmp_1@GRAD', 'concat_0.tmp_0@GRAD'
                    ],
                })

                train_start_time = time.time()
                train_end_time = time.time()
                logger.info("Train Dataset Done, using time {} second.".format(
                    train_end_time - train_start_time))

                logger.info("Pass: {}, Running Dataset Begin.".format(index))
                fetch_info = [
                    "Pass: {} Var {}".format(index, var_name)
                    for var_name in self.metrics
                ]
                fetch_vars = [var for _, var in self.metrics.items()]
                print_step = int(config.get("runner.print_interval"))
                self.exe.train_from_dataset(
                    program=paddle.static.default_main_program(),
                    dataset=dataset,
                    fetch_list=fetch_vars,
                    fetch_info=fetch_info,
                    print_period=print_step,
                    debug=config.get("runner.dataset_debug"))
                dataset.release_memory()
                global_auc = self.get_global_auc()
                logger.info(" global auc %f" % global_auc)

                metric_list = list(self.model.auc_stat_list) + list(
                    self.model.metric_list)

                metric_types = ["int64"] * len(self.model.auc_stat_list) + [
                    "float32"
                ] * len(self.model.metric_list)

                metric_str = self.get_global_metrics_str(
                    fluid.global_scope(), metric_list, "update pass:")

                logger.info(" global metric %s" % metric_str)

                self.clear_metrics(fluid.global_scope(), metric_list,
                                   metric_types)
                if fleet.is_first_worker():
                    if index % self.checkpoint_per_pass == 0:
                        self.save_model(save_model_path, day, index)
                        self.write_model_donefile(
                            output_path=save_model_path,
                            day=day,
                            pass_id=index,
                            xbox_base_key=model_base_key,
                            hadoop_fs_name=self.hadoop_fs_name,
                            hadoop_fs_ugi=self.hadoop_fs_ugi)

                    if index % self.save_delta_frequency == 0:
                        last_xbox_day, last_xbox_pass, last_xbox_path, _ = self.get_last_save_xbox(
                            save_model_path, self.hadoop_fs_name,
                            self.hadoop_fs_ugi)
                        if int(day) < last_xbox_day or int(
                                day) == last_xbox_day and int(
                                    index) <= last_xbox_pass:
                            log_str = "delta model exists"
                            logger.info(log_str)
                        else:
                            self.save_xbox_model(save_model_path, day, index,
                                                 self.hadoop_fs_name,
                                                 self.hadoop_fs_ugi)  # 1 delta
                            self.write_xbox_donefile(
                                output_path=save_model_path,
                                day=day,
                                pass_id=index,
                                model_base_key=model_base_key,
                                hadoop_fs_name=self.hadoop_fs_name,
                                hadoop_fs_ugi=self.hadoop_fs_ugi)
                fleet.barrier_worker()

            if fleet.is_first_worker():
                last_base_day, last_base_path, last_base_key = self.get_last_save_xbox_base(
                    save_model_path, self.hadoop_fs_name, self.hadoop_fs_ugi)
                logger.info(
                    "one epoch finishes, get_last_save_xbox, last_base_day = {}, last_base_path = {}, last_base_key = {}".
                    format(last_base_day, last_base_path, last_base_key))
                next_day = int(self.get_next_day(day))
                if next_day <= last_base_day:
                    model_base_key = last_base_key
                else:
                    model_base_key = int(time.time())
                    fleet.shrink(10)
                    self.save_xbox_model(save_model_path, next_day, -1,
                                         self.hadoop_fs_name,
                                         self.hadoop_fs_ugi)
                    self.write_xbox_donefile(
                        output_path=save_model_path,
                        day=next_day,
                        pass_id=-1,
                        model_base_key=model_base_key,
                        hadoop_fs_name=self.hadoop_fs_name,
                        hadoop_fs_ugi=self.hadoop_fs_ugi)
                    self.save_batch_model(save_model_path, next_day)
                    self.write_model_donefile(
                        output_path=save_model_path,
                        day=next_day,
                        pass_id=-1,
                        xbox_base_key=model_base_key,
                        hadoop_fs_name=self.hadoop_fs_name,
                        hadoop_fs_ugi=self.hadoop_fs_ugi)

            day = self.get_next_day(day)


if __name__ == "__main__":
    paddle.enable_static()
    parser = argparse.ArgumentParser("PaddleRec train script")
    parser.add_argument(
        '-m',
        '--config_yaml',
        type=str,
        required=True,
        help='config file path')
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    yaml_helper = YamlHelper()
    config = yaml_helper.load_yaml(args.config_yaml)
    config["yaml_path"] = args.config_yaml
    config["config_abs_dir"] = args.abs_dir
    trainer = Training(config)
    trainer.run()
