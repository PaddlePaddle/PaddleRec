from __future__ import print_function

import random
import numpy as np
from pathlib import Path
from utils.static_ps.reader_helper import get_reader, get_example_num, get_file_list, get_word_num
from utils.static_ps.program_helper import get_model, get_strategy
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
        self.hadoop_fs_name = config.get("runner.hadoop_fs_name")
        self.hadoop_fs_ugi = config.get("runner.hadoop_fs_ugi")
        self.reader_type = "QueueDataset"
        self.train_local = self.hadoop_fs_name is None or self.hadoop_fs_ugi is None
        self.split_interval = config.get("runner.split_interval")
        self.split_per_pass = config.get("runner.split_per_pass")
        self.save_delta_frequency = config.get("runner.save_delta_frequency",6)
        self.checkpoint_per_pass = config.get("runner.checkpoint_per_pass",6)
        self.save_first_base = config.get("runner.save_first_base")
        self.days = config.get("runner.days")
        self.hours = config.get("runner.hours")
        print("self.days ",self.days)
        print("self.hours ",self.hours)
        print("self.data = ",config.get("runner.train_data_dir"))

    def run(self):
        os.environ["PADDLE_WITH_GLOO"] = "1"
        role = role_maker.PaddleCloudRoleMaker(init_gloo = True)
        fleet.init(role)
        self.init_network()
        if fleet.is_server():
            self.run_server()
        elif fleet.is_worker():
            self.run_worker()
            fleet.stop_worker()
        logger.info("successfully completed running, Exit.")
        # self.init_network()
        # self.run_worker()

    def get_next_day(day):
        return os.popen('date -d"%s' % day + ' +1 days" +"%Y%m%d"').read().strip()

    def save_fleet_model(self, path, mode=0):
        fleet.save_persistables(None, path, mode=mode)

    def save_model(self, output_path, day, pass_id):
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
        self.save_fleet_model(model_path)

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
            content  = "%s\t%lu\t%s\t%s\t%d" % (day, xbox_base_key,\
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
                if not file.is_file():
                    with open(donefile_path, "w") as f:
                        f.write(content + "\n")
                    return
                with open(donefile_path,encoding='utf-8') as f:
                    pre_content = f.read()
                exist = False
                lines = pre_content.split("\n")
                if len(lines) > 0 and lines[-1] == "":
                    lines = lines[:-1]
                day_list = [i.split("\t")[0] for i in lines]
                pass_list = [i.split("\t")[3] for i in lines]
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
                    logger.info("write %s/%s %s succeed" % \
                                (day, pass_id, donefile_name))
                else:
                    logger.info("not write %s because %s/%s already "
                                "exists" % (donefile_name, day, pass_id))

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
        if scope.find_var(stat_pos) is None or scope.find_var(stat_neg) is None:
            self.rank0_print("not found auc bucket")
            return None
        fleet.barrier_worker()
        # auc pos bucket
        pos = np.array(scope.find_var(stat_pos).get_tensor())
        # auc pos bucket shape
        old_pos_shape = np.array(pos.shape)
        # reshape to one dim
        pos = pos.reshape(-1)
        global_pos = np.copy(pos) * 0
        # mpi allreduce
        fleet.util.all_reduce(pos)
        # reshape to its original shape
        global_pos = global_pos.reshape(old_pos_shape)

        # auc neg bucket
        neg = np.array(scope.find_var(stat_neg).get_tensor())
        old_neg_shape = np.array(neg.shape)
        neg = neg.reshape(-1)
        global_neg = np.copy(neg) * 0
        fleet.util.all_reduce(neg)
        global_neg = global_neg.reshape(old_neg_shape)

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

        auc_value = None
        if pos * neg == 0 or total_ins_num == 0:
            auc_value = 0.5
        else:
            auc_value = area / (pos * neg)

        fleet.barrier_worker()
        return auc_value

    def save_xbox_model(self, output_path,day,pass_id,hadoop_fs_name,hadoop_fs_ugi,hadoop_home="$HADOOP_HOME"):
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
                            self.predict) #0 save checkpoints
        else:
            model_name = "inference_model"
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
                client.makedirs(dest)

            client.upload(model_name, dest, multi_processes=5, overwrite=True)
        fleet.save_persistables(executor=self.exe,dirname=model_path,mode=mode)
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

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              last_save_day, last_save_pass, last_path, xbox_base_key = \
                  fleet_util.get_last_save_model("hdfs:/my/path", 20190722, 88)

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
            pre_content = client.cat(donefile_path)
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

    def get_online_pass_interval(self, days, hours, split_interval,
                                 split_per_pass, is_data_hourly_placed):
        """
        get online pass interval

        Args:
            days(str): days to train
            hours(str): hours to train
            split_interval(int|str): split interval
            split_per_pass(int}str): split per pass
            is_data_hourly_placed(bool): is data hourly placed

        Returns:
            online_pass_interval(list)

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              online_pass_interval = fleet_util.get_online_pass_interval(
                  days="{20190720..20190729}",
                  hours="{0..23}",
                  split_interval=5,
                  split_per_pass=2,
                  is_data_hourly_placed=False)

        """
        days = os.popen("echo -n " + days).read().split(" ")
        hours = os.popen("echo -n " + hours).read().split(" ")
        split_interval = int(split_interval)
        split_per_pass = int(split_per_pass)
        splits_per_day = 24 * 60 // split_interval
        pass_per_day = splits_per_day // split_per_pass
        left_train_hour = int(hours[0])
        right_train_hour = int(hours[-1])

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

        return days, hours, online_pass_interval

    def write_xbox_donefile(self,
                            output_path,
                            day,
                            pass_id,
                            model_base_key,
                            hadoop_fs_name,
                            hadoop_fs_ugi,
                            hadoop_home="$HADOOP_HOME",
                            donefile_name=None):
        print("in write_xbox_donefile day = ",day, " pass-id = ",pass_id)
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
            xbox_str = self._get_xbox_str(model_path=model_path, xbox_base_key=xbox_base_key,
                                          mode=mode)
            print("xbox str",xbox_str)
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
                            output_path,
                            donefile_name,
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
                        output_path,
                        donefile_name,
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
                with open(donefile_path,encoding='utf-8') as f:
                    pre_content = f.read()
                exist = False
                last_line = pre_content.split("\n")[-1]
                if last_line == '':
                    last_line = pre_content.split("\n")[-2]
                last_dict = json.loads(last_line,strict=False)
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
                      hadoop_fs_name = None,
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
        xbox_dict["input"] = ("" if hadoop_fs_name is None else hadoop_fs_name) + model_path.rstrip("/") + "/000"
        return json.dumps(xbox_dict)

    def init_network(self):
        self.model = get_model(self.config)
        self.input_data = self.model.create_feeds()
        self.inference_feed_var = self.model.create_feeds(is_infer=True)
        self.metrics = self.model.net(self.input_data)
        self.inference_target_var = self.model.inference_target_var
        self.predict = self.model.predict
        logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))
        self.model.create_optimizer(get_strategy(self.config))


    def run_server(self):
        logger.info("Run Server Begin")
        fleet.init_server(config.get("runner.warmup_model_path","./warmup"))
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
            hdfs_client = HDFSClient("$HADOOP_HOME", configs)
            for i in path_array:
                cur_path = hdfs_client.ls(i)
                if self.hadoop_fs_name.startswith("hdfs:"):
                    cur_path = ["hdfs:" + j for j in cur_path]
                elif self.hadoop_fs_name.startswith("afs:"):
                    cur_path = ["hdfs:" + j for j in cur_path]
                result += cur_path
        return result

    def prepare_dataset(self, day, pass_index):
        #dataset, file_list = get_reader(self.input_data, config)

        dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_use_var(self.input_data)
        dataset.set_batch_size(self.config.get('runner.train_batch_size'))
        dataset.set_thread(self.config.get('runner.train_thread_num',1))
        data_path = self.config.get("runner.train_data_dir")
        if not self.train_local:
            dataset.set_hdfs_config(self.hadoop_fs_name,self.hadoop_fs_ugi)
        cur_path = []
        for i in self.online_intervals[pass_index-1]:
            cur_path.append(data_path.rstrip("/") + "/" + day + "/" + i)
        global_file_list = self.file_ls(cur_path)
        print("global_file_list",global_file_list)
        my_file_list = fleet.util.get_file_shard(global_file_list)
        print("my_file_list",my_file_list)
        dataset.set_filelist(my_file_list)
        pipe_command = self.config.get("runner.pipe_command")
        dataset.set_pipe_command(self.config.get("runner.pipe_command"))
        utils_path = common.get_utils_file_path()
        print("utils_path: {}".format(utils_path))
        dataset.set_pipe_command("{} {} {}".format(pipe_command,
                                              config.get("yaml_path"),
                                              utils_path))
        dataset.load_into_memory()
        return dataset

    def run_worker(self):
        logger.info("Run Online Worker Begin")
        use_cuda = int(config.get("runner.use_gpu"))
        place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
        self.exe = paddle.static.Executor(place)
        self.exe.run(paddle.static.default_startup_program())
        fleet.init_worker()
        #self.split_trainfile()
        print("self.days ",self.days)
        print("self.hours ",self.hours)
        days, hours, self.online_intervals = self.get_online_pass_interval(self.days,self.hours,self.split_interval,self.split_per_pass,False)
        self.save_model_path = self.config.get("runner.model_save_path")
        self.warm_start_model_path = config.get("runner.warm_start_model_path","./warmup")
        #donefile_path = self.config.get("runner.donefile_path","./done")
        if self.save_model_path and (not os.path.exists(self.save_model_path)):
            os.makedirs(self.save_model_path)

        last_day, last_pass, last_path, model_base_key = self.get_last_save_model(self.save_model_path, self.hadoop_fs_name, self.hadoop_fs_ugi)
        if last_day != -1:
            fleet.init_server(last_path, mode=0)

        save_first_base = self.save_first_base
        for i in range(len(days)):
            day = days[i]
            if last_day != -1 and int(day) < last_day:
                i+= 1
                continue
            #base_model_saved = False
            save_model_path = self.save_model_path
            for pass_id in range(len(self.online_intervals)):
                if (last_day != -1 and int(day) == last_day) and (last_pass != -1 and int(index) < last_pass):
                    #base_model_saved = True
                    continue
                if fleet.is_first_worker() and save_first_base:
                    save_first_base = False
                    last_base_day, last_base_path, tmp_xbox_base_key = \
                        self.get_last_save_xbox_base(save_model_path, self.hadoop_fs_name, self.hadoop_fs_ugi)
                    if int(day) > last_base_day:
                        model_base_key = int(time.time())
                        self.save_xbox_model(save_model_path,day,-1)
                        self.write_xbox_donefile(output_path=save_model_path,day=day,pass_id=-1,model_base_key=model_base_key,hadoop_fs_name=self.hadoop_fs_name,hadoop_fs_ugi=self.hadoop_fs_ugi)
                    elif int(day) == last_base_day:
                        model_base_key = tmp_xbox_base_key

                # if not base_model_saved:
                #     self.save_xbox_model(save_model_path,day,-1)
                #     self.write_xbox_donefile(output_path=save_model_path,day=day,pass_id=-1,model_base_key=model_base_key,hadoop_fs_name=self.hadoop_fs_name,hadoop_fs_ugi=self.hadoop_fs_ugi)
                #     base_model_saved = True

                print("new day ",day," new pass ",pass_id)
                index = pass_id + 1
                prepare_data_start_time = time.time()
                dataset = self.prepare_dataset(day, index)
                prepare_data_end_time = time.time()
                logger.info(
                    "Prepare Dataset Done, using time {} second.".format(
                        prepare_data_end_time - prepare_data_start_time))

                train_start_time = time.time()
                train_end_time = time.time()
                logger.info(
                        "Train Dataset Done, using time {} second.".format(train_end_time - train_start_time))

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
                logger.info(" global auc ",global_auc)
                if fleet.is_first_worker():
                    if index % self.checkpoint_per_pass == 0:
                        self.save_model(save_model_path,day,index)
                        self.write_model_donefile(output_path=save_model_path,
                                              day=day,
                                              pass_id=index,
                                              xbox_base_key=model_base_key,
                                              hadoop_fs_name=self.hadoop_fs_name,
                                            hadoop_fs_ugi=self.hadoop_fs_ugi)

                    if index % self.save_delta_frequency == 0:
                        last_xbox_day, last_xbox_pass, last_xbox_path, _ = self.get_last_save_xbox(save_model_path,self.hadoop_fs_name,self.hadoop_fs_ugi)
                        if int(day) < last_xbox_day or int(day) == last_xbox_day and int(index) <= last_xbox_pass:
                            log_str = "delta model exists"
                            logger.info(log_str)
                        else:
                            self.save_xbox_model(save_model_path, day, index) #1 delta
                            self.write_xbox_donefile(output_path=save_model_path, day=day, pass_id=index,
                                                 model_base_key=model_base_key, hadoop_fs_name=self.hadoop_fs_name,
                                                 hadoop_fs_ugi=self.hadoop_fs_ugi)
                fleet.barrier_worker()


            if fleet.is_first_worker():
                last_base_day, last_base_path, last_base_key = self.get_last_save_xbox_base(save_model_path, self.hadoop_fs_name,self.hadoop_fs_ugi)
                next_day = int(self.get_next_day(day))
                if next_day <= last_base_day:
                    model_base_key = last_base_key
                else:
                    model_base_key = int(time.time())
                    fleet.shrink_sparse_table()
                    self.save_xbox_model(save_model_path, day, -1)
                    self.write_xbox_donefile(output_path=save_model_path, day=day, pass_id=-1,
                                             model_base_key=model_base_key, hadoop_fs_name=self.hadoop_fs_name,
                                             hadoop_fs_ugi=self.hadoop_fs_ugi)
                    # else:
                    #
                    #     self.save_xbox_model(save_model_path,0,index)  #1 delta
                    #     print("save delta ",index)
                    #     self.write_xbox_donefile(output_path=save_model_path,day=0, pass_id=index, model_base_key=model_base_key,hadoop_fs_name=self.hadoop_fs_name,hadoop_fs_ugi=self.hadoop_fs_ugi)

        # fleet.barrier_worker()
        # fleet.save_inference_model(
        #                     self.exe, self.warm_start_model_path,
        #                     [feed.name for feed in self.inference_feed_var],
        #                     self.inference_target_var,
        #                     mode=0) #0 save checkpoints


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
