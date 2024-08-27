# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Global Utilities Functions
"""
import os
import sys
import json
import time
import math
import shutil
import glob
import re
import traceback
import collections
import numpy as np
import pickle as pkl
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta

import paddle
import paddle.static as static
import paddle.base.core as core
import paddle.distributed.fleet as fleet
from pgl.utils.logger import log

import util_hadoop as HFS

def allreduce_min(input_num):
    """
    mpi reduce min
    """
    return fleet.all_reduce(input_num, mode="min")

def barrier():
    """
    mpi barrier
    """
    return fleet.barrier_worker()

def get_global_value(value_sum, value_cnt):
    """ get global value """
    value_sum = np.array(paddle.static.global_scope().find_var(value_sum.name)
                         .get_tensor())
    value_cnt = np.array(paddle.static.global_scope().find_var(value_cnt.name)
                         .get_tensor())
    return value_sum / np.maximum(value_cnt, 1)


def get_batch_num(value_cnt):
    """ get global value """
    value_cnt = np.array(paddle.static.global_scope().find_var(value_cnt.name).get_tensor())
    return value_cnt


def save_holder_names(model_dict, filename):
    """ save holder names """
    holder_name_list = [str(var.name) for var in model_dict.holder_list]
    with open(filename, "w") as f:
        for name in holder_name_list:
            f.write("%s\n" % name)


def parse_path(path):
    """
    Args:
        path: path has follow format:
            1, /your/local/path
            2, afs:/your/remote/afs/path
            3, hdfs:/your/remote/hdfs/path
            4, afs://xxx.baidu.com:xxxx/your/remote/afs/path
            5, hdfs://xxx.baidu.com:xxxx/your/remote/hdfs/path

    Return:
        mode: 3 different modes: local, afs, hdfs
        output_path: /your/lcoal_or_remote/path

    """
    if path.startswith("afs"):
        mode = "afs"
        if "baidu.com" in path:
            # 4, afs://xxx.baidu.com:xxxx/your/remote/afs/path
            output_path = remove_prefix_of_hadoop_path(path)
        else:
            # 2, afs:/your/remote/afs/path
            output_path = path.split(':')[-1]

    elif path.startswith("hdfs"):
        mode = "hdfs"
        if "baidu.com" in path:
            #  5, hdfs://xxx.baidu.com:xxxx/your/remote/hdfs/path
            output_path = remove_prefix_of_hadoop_path(path)
        else:
            # 3, hdfs:/your/remote/hdfs/path
            output_path = path.split(':')[-1]
    else:
        mode = "local"
        output_path = path

    return mode, output_path


def remove_prefix_of_hadoop_path(hadoop_path):
    """
    Args:
        hadoop_path: afs://xxx.baidu.com:xxxx/your/remote/hadoop/path

    Return:
        output_path: /your/remote/hadoop/path
    """
    output_path = hadoop_path.split(":")[-1]
    output_path = re.split("^\d+", output_path)[-1]
    return output_path

def set_hadoop_account(hadoop_bin, fs_name, fs_ugi):
    HFS.set_hadoop_account(hadoop_bin, fs_name, fs_ugi)

def load_pretrained_model(exe, model_dict, args, model_path):
    """ load pretrained model """
    # multi node
    if paddle.distributed.get_world_size() > 1:
        sparse_model_path = os.path.join(model_path, "%03d" % (paddle.distributed.get_rank()))
    else:
        sparse_model_path = model_path
    if os.path.exists(sparse_model_path): # local directory
        sparse_params_path = os.path.join(sparse_model_path, "000")
        if os.path.exists(sparse_params_path):
            log.info("[WARM] load sparse model from %s" % sparse_params_path)
            if "train_storage_mode" in args and args.train_storage_mode == "SSD_EMBEDDING":
                if "load_binary_mode" in args and args.load_binary_mode is True:
                    # in train_storage_mode=SSD_EMBEDDING, mode=0 means load binary batch_model,
                    # mode=4 means save origin batch_model
                    fleet.load_model(sparse_model_path, mode=0)
                else:
                    fleet.load_model(sparse_model_path, mode=4)
            else:
                fleet.load_model(sparse_model_path, mode=0)
            log.info("[WARM] load sparse model from %s finished." % sparse_params_path)
        else:
            raise Exception("[ERROR] sparse model [%s] is not existed" % sparse_params_path)

        dense_params_path = os.path.join(model_path, "dense_vars")

    else:   # load from hadoop path
        mode, sparse_model_path = parse_path(sparse_model_path)
        sparse_model_path = HFS.check_hadoop_path(sparse_model_path)
        sparse_params_path = os.path.join(sparse_model_path, "000")

        if HFS.exists(sparse_params_path):
            log.warning("Downloading sparse model: %s -> %s" % (sparse_params_path, "./"))
            user, passwd = args.fs_ugi.split(',')
            sparse_params_path = sparse_params_path.replace(args.fs_name, '')
            gzshell_download(args.fs_name, user, passwd, sparse_params_path, "./")
            log.info("Loading sparse model from 000")
            if "train_storage_mode" in args and args.train_storage_mode == "SSD_EMBEDDING":
                if "load_binary_mode" in args and args.load_binary_mode is True:
                    fleet.load_model("000", mode=0)
                else:
                    fleet.load_model("000", mode=4)
            else:
                fleet.load_model("000", mode=0)
            log.info("[WARM] loaded sparse model from %s" % sparse_params_path)
        else:
            raise Exception("[ERROR] sparse model [%s] is not existed" % sparse_params_path)

        hadoop_dense_params_path = os.path.join(model_path, "dense_vars")
        mode, hadoop_dense_params_path = parse_path(hadoop_dense_params_path)
        hadoop_dense_params_path = HFS.check_hadoop_path(hadoop_dense_params_path)

        dense_params_path = "./dense_vars"

        if HFS.exists(hadoop_dense_params_path):
            if os.path.exists(dense_params_path):
                run_cmd("rm -rf %s" % dense_params_path)
            log.warning("Downloading dense model: %s -> %s" % (hadoop_dense_params_path, "./"))
            user, passwd = args.fs_ugi.split(',')
            hadoop_dense_params_path = hadoop_dense_params_path.replace(args.fs_name, '')
            gzshell_download(args.fs_name, user, passwd, hadoop_dense_params_path, "./")
        else:
            log.info("[WARM] dense_model [%s] is not existed" % hadoop_dense_params_path)

    # load dense vars
    if os.path.exists(dense_params_path):
        log.info("[WARM] loading dense parameters from: %s" % dense_params_path)
        all_vars = model_dict.train_program.global_block().vars
        for filename in os.listdir(dense_params_path):
            if filename in all_vars:
                log.info("[WARM] var %s existed" % filename)
            else:
                log.info("[WARM_MISS] var %s not existed" % filename)

        paddle.static.io.load_vars(exe,
               dense_params_path,
               model_dict.train_program,
               predicate=name_not_have_sparse)
        log.info("Loaded dense parameters from: %s" % dense_params_path)
    elif args.pretrained_model:
        # if hadoop model path did not include dense params, then load dense pretrained_model from dependency
        dependency_path = os.getenv("DEPENDENCY_HOME") # see env_run/scripts/train.sh for details
        dense_path = os.path.join(dependency_path, args.pretrained_model)
        log.info("[WARM] loading dense parameters from: %s" % dense_path)
        paddle.static.set_program_state(model_dict.train_program, model_dict.state_dict)
        log.info("[WARM] loaded dense parameters from: %s" % dense_path)
    else:
        log.info("[WARM] dense model is not existed, skipped")

    return 0


def save_pretrained_model(exe, save_path, args, mode="hdfs", save_mode=0):
    """save pretrained model"""
    if mode == "hdfs":
        save_path = HFS.check_hadoop_path(save_path)
        HFS.rm(save_path)

    if "train_storage_mode" in args and args.train_storage_mode == "SSD_EMBEDDING":
        if "save_binary_mode" in args and args.save_binary_mode is True:
            # in train_storage_mode="SSD_EMBEDDING", mode=3 means save binary batch_model,
            # mode=7 means save origin batch_model
            fleet.save_persistables(exe, save_path, mode=save_mode)
        else:
            fleet.save_persistables(exe, save_path, mode=save_mode + 4)
    else:
        # in train_storage_mode="MEM_EMBEDDING",
        # mode=3 means save batch_model(unseenday+1),
        # mode=0 means save checkpoint model
        fleet.save_persistables(exe, save_path, mode=save_mode)


def name_not_have_sparse(var, local_param = None):
    """
    persistable var which not contains pull_box_sparse
    """
    res = "sparse" not in var.name and \
            paddle.static.io.is_persistable(var) and \
            var.name != "embedding" and \
            "learning_rate" not in var.name and \
            "_generated_var" not in var.name and \
           (local_param is None or var.name in local_param)
    return res


def remove_path(path):
    """
        remove file or path
    """
    try:
        if not os.path.isdir(path):
            os.remove(path)
            return 1
        cnt = 0
        dirs = os.listdir(path)
        for name in dirs:
            full_path = os.path.join(path, name)
            if os.path.isdir(full_path):
                ret = remove_path(full_path)
                if ret > 0:
                    cnt = cnt + ret
            else:
                os.remove(full_path)
        os.rmdir(path)
        return cnt
    except Exception as e:
        log.info('remove_path %s exception: %s' % (path, e))
        return 1


def save_model(exe, model_dict, args, local_model_path, model_save_path, local_param=[], save_mode=0):
    """final save model"""
    mode, model_save_path = parse_path(model_save_path)
    _, working_root = parse_path(args.working_root)
    # multi node
    if paddle.distributed.get_world_size() > 1:
        working_root = os.path.join(working_root, "model")

    # save sparse table
    log.info("save sparse table")
    if mode == "hdfs":
        save_pretrained_model(exe, model_save_path, args, mode = "hdfs", save_mode=save_mode)
        if os.path.exists(local_model_path):
            run_cmd("rm -rf %s" % local_model_path)
        log.info("delete local model: %s" % local_model_path)
    elif mode == "afs":
        save_pretrained_model(exe, local_model_path, args, mode = "local", save_mode=save_mode)
        user, passwd = args.fs_ugi.split(',')
        log.info("being to upload model to: %s " % model_save_path)
        #  HFS.rm(model_save_path)
        gzshell_upload(args.fs_name, user, passwd, local_model_path, "afs:%s" % working_root)
        log.info("model has been saved, model_path: %s" % model_save_path)
        if os.path.exists(local_model_path):
            run_cmd("rm -rf %s" % local_model_path)
        log.info("delete local model: %s" % local_model_path)
    else:
        save_pretrained_model(exe, local_model_path, args, mode = "local", save_mode=save_mode)
        if paddle.distributed.get_world_size() is 1:
            model_root = working_root
        else:
            model_root = working_root + "/model/"
        if os.path.exists(model_root):
            remove_path(model_root)
        make_dir(working_root)
        tmp_path = os.path.join(working_root, "model")
        if os.path.exists(tmp_path):
            run_cmd("rm -rf %s" % (tmp_path))
        run_cmd("mv %s %s" % (local_model_path, working_root))
        log.info("model has been saved in local path: %s" % working_root)

    is_sharding = False
    if getattr(args, "sharding", None):
        is_sharding = True
    # master node  and not thread sharing
    if not fleet.worker_index() is 0 and is_sharding is False:
        return 0
    name_suffix = ""
    if is_sharding is True:
        name_suffix = "_" + str(fleet.worker_index())

    # save dense model
    log.info("[SAVE] save dense model")
    local_var_save_path = "./dense_vars"
    if os.path.exists(local_var_save_path):
        shutil.rmtree(local_var_save_path)

    if is_sharding is False:
        paddle.static.io.save_vars(exe,
            local_var_save_path,
            model_dict.train_program,
            predicate=name_not_have_sparse)
    else:
        gpu_num = len(local_param)
        for i in range(gpu_num -1 , -1, -1):
            place = paddle.CUDAPlace(i)
            exe_temp = static.Executor(place)
            paddle.static.io.save_vars(exe_temp,
                local_var_save_path,
                model_dict.train_program,
                predicate = lambda var : name_not_have_sparse(var, local_param[i]))
            exe_temp.close()

    # local_var_save_path is not existed if no variable in model
    if os.path.exists(local_var_save_path):
        if mode == "hdfs" or mode == "afs":
            dense_save_path = HFS.check_hadoop_path(os.path.join(model_save_path, "dense_vars"))
            HFS.rm(dense_save_path)
            ret = HFS.put_files(local_var_save_path, dense_save_path)
            if ret != 0:
                log.warning("Fail to upload dense model: %s -> %s" \
                    % (local_var_save_path, model_save_path))
                log.warning("[HADOOP] you can check out the log in [env_run/src/hadoop_err.log]")
                return -1
        else:
            run_cmd("mv %s %s" % (local_var_save_path, model_save_path))

    if hasattr(model_dict, "sr_model"):
        log.info("[SAVE] saving erniesage state_dict (dense model) for export inference model")
        params = []
        for param in local_param:
            params = params + param
        save_erniesage_state_dict(model_dict, mode, model_save_path, params, name_suffix)

    return 0


def save_erniesage_state_dict(model_dict, mode, model_save_path, local_param, name_suffix):
    """ doc """
    local_var_save_path = "./dense_vars_for_export"
    if os.path.exists(local_var_save_path):
        shutil.rmtree(local_var_save_path)

    tensor_state_dict = {}
    for key, item in model_dict.sr_model.state_dict().items():
        if item.name in local_param:
            item = item.get_value()
            tensor_state_dict[key] = item
    paddle.save(tensor_state_dict,
            os.path.join(local_var_save_path, "state_dict.pdparams" + name_suffix))
    #  paddle.static.save(model_dict.train_program, local_var_save_path)
    #  paddle.jit.save(model_dict.sr_model, os.path.join(local_var_save_path, "erniesage"))

    if mode == "hdfs" or mode == "afs":
        HFS.rm(os.path.join(model_save_path, "dense_vars_for_export"))
        ret = HFS.put_files(local_var_save_path, os.path.join(model_save_path, "dense_vars_for_export"))
        if ret != 0:
            log.warning("Fail to upload erniesage_state_dict: %s -> %s" \
                    % (local_var_save_path, model_save_path))
            return -1
    else:
        run_cmd("mv %s %s" % (local_var_save_path, model_save_path))

    return 0


def upload_embedding(args, local_embed_path):
    """ doc """
    mode, infer_result_path = parse_path(args.infer_result_path)
    _, working_root = parse_path(args.working_root)
    # multi node
    if paddle.distributed.get_world_size() > 1:
        working_root = os.path.join(working_root, "embedding")

    if mode == "hdfs":
        HFS.rm(infer_result_path)
        HFS.mkdir(infer_result_path)

        log.info("being to upload embedding to: %s " % infer_result_path)
        for file in glob.glob(os.path.join(local_embed_path, "*")):
            basename = os.path.basename(file)
            ret = HFS.put(file, infer_result_path)
            if ret != 0:
                log.warning("Fail to upload embedding: %s -> %s" \
                    % (file, infer_result_path))
                return -1
        log.info("[hadoop put] embedding has been upload to: %s " % infer_result_path)
        if os.path.exists(local_embed_path):
            run_cmd("rm -rf %s" % local_embed_path)
        log.info("delete local embedding: %s" % local_embed_path)

    elif mode == "afs":
        log.info("being to upload embedding to: %s " % infer_result_path)
        #  HFS.rm(infer_result_path)
        user, passwd = args.fs_ugi.split(',')
        gzshell_upload(args.fs_name, user, passwd, local_embed_path, "afs:%s" % working_root)
        log.info("[gzshell] embedding has been upload to: %s " % infer_result_path)
        if os.path.exists(local_embed_path):
            run_cmd("rm -rf %s" % local_embed_path)
        log.info("delete local embedding: %s" % local_embed_path)
    else:
        make_dir(working_root)
        tmp_path = os.path.join(working_root, "embedding")
        if os.path.exists(tmp_path):
            run_cmd("rm -rf %s" % (tmp_path))
        run_cmd("mv %s %s" % (local_embed_path, working_root))
        log.info("embedding has been saved in local path: %s" % working_root)

    return 0


def upload_dump_walk(args, local_dump_path):
    mode, dump_save_path = parse_path(args.dump_walk_path)
    _, working_root = parse_path(args.working_root)
    # multi node
    if paddle.distributed.get_world_size() > 1:
        working_root = os.path.join(working_root, "dump_walk")
    if mode == "hdfs":
        HFS.rm(dump_save_path)
        HFS.mkdir(dump_save_path)

        log.info("being to upload walk_path to: %s " % dump_save_path)
        for file in glob.glob(os.path.join(local_dump_path, "*")):
            basename = os.path.basename(file)
            ret = HFS.put(file, dump_save_path)
            if ret != 0:
                log.warning("Fail to upload walk_path: %s -> %s" \
                    % (file, dump_save_path))
                return -1
        log.info("[hadoop put] walk_path has been upload to: %s " % dump_save_path)

    elif mode == "afs":
        log.info("being to upload walk_path to: %s " % dump_save_path)
        #  HFS.rm(dump_save_path)
        user, passwd = args.fs_ugi.split(',')
        gzshell_upload(args.fs_name, user, passwd, local_dump_path, "afs:%s" % working_root)
        log.info("[gzshell] walk_path has been upload to: %s " % dump_save_path)
    else:
        make_dir(working_root)
        run_cmd("mv %s %s" % (local_dump_path, working_root))
        log.info("walk_path has been saved in local path: %s" % working_root)

    return 0


def upload_dump_neighbors(args, local_dump_path):
    mode, dump_save_path = parse_path(args.dump_neighbors_path)
    _, working_root = parse_path(args.working_root)
    # multi node
    if paddle.distributed.get_world_size() > 1:
        working_root = os.path.join(working_root, "dump_neighbors")
    if mode == "hdfs":
        HFS.rm(dump_save_path)
        HFS.mkdir(dump_save_path)

        log.info("being to upload neighbors to: %s " % dump_save_path)
        for file in glob.glob(os.path.join(local_dump_path, "*")):
            basename = os.path.basename(file)
            ret = HFS.put(file, dump_save_path)
            if ret != 0:
                log.warning("Fail to upload neighbors: %s -> %s" \
                    % (file, dump_save_path))
                return -1
        log.info("[hadoop put] neighbors has been upload to: %s " % dump_save_path)

    elif mode == "afs":
        log.info("being to upload walk_path to: %s " % dump_save_path)
        #  HFS.rm(dump_save_path)
        user, passwd = args.fs_ugi.split(',')
        gzshell_upload(args.fs_name, user, passwd, local_dump_path, "afs:%s" % working_root)
        log.info("[gzshell] neighbors has been upload to: %s " % dump_save_path)
    else:
        make_dir(working_root)
        run_cmd("mv %s %s" % (local_dump_path, working_root))
        log.info("neighbors has been saved in local path: %s" % working_root)

    return 0


def hadoop_touch_done(path):
    """ touch hadoop done """
    if fleet.worker_index() == 0:
        with open("to.hadoop.done", 'w') as f:
            f.write("infer done\n")
        ret = HFS.put("to.hadoop.done", os.path.join(path, "to.hadoop.done"))
        if ret != 0:
            log.warning("Fail to upload hadoop done: %s -> %s" \
                    % ("to.hadoop.done", os.path.join(path, "to.hadoop.done")))
            return -1

    return 0


def get_job_info():
    """ get job info """
    info_list = []
    time_msg = "%s" % datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    info_list.append(time_msg)
    info_list.append("user_name: " + str(os.getenv("SYS_USER_NAME")))
    info_list.append("job_name: " + str(os.getenv("SYS_JOB_NAME")))
    info_list.append("job_id: " + str(os.getenv("SYS_JOB_ID")))
    info_list.append("code path: " + str(os.getenv("CODE_URI")))
    info_list.append("fs_name: " + str(os.getenv("FS_NAME")))
    info_list.append("fs_ugi: " + str(os.getenv("FS_UGI")))
    info = "\n".join(info_list)
    return info + "\n"


def print_useful_info():
    """ print useful info """
    try:
        import socket
        ip_addres = socket.gethostbyname(socket.gethostname())
        log.info("The IP_ADDRESS of this machine is: %s" % ip_addres)
    except Exception as e:
        log.info("%s" % (e))
        log.info("can not import socket")


# Global error handler
def global_except_hook(exctype, value, traceback):
    """global except hook"""
    import sys
    try:
        import mpi4py.MPI
        sys.stderr.write("\n*****************************************************\n")
        sys.stderr.write("Uncaught exception was detected on rank {}. \n".format(
            mpi4py.MPI.COMM_WORLD.Get_rank()))
        from traceback import print_exception
        print_exception(exctype, value, traceback)
        sys.stderr.write("*****************************************************\n\n\n")
        sys.stderr.write("\n")
        sys.stderr.write("Calling MPI_Abort() to shut down MPI processes...\n")
        sys.stderr.flush()
    finally:
        try:
            import mpi4py.MPI
            mpi4py.MPI.COMM_WORLD.Abort(1)
        except Exception as e:
            sys.stderr.write("*****************************************************\n")
            sys.stderr.write("Sorry, we failed to stop MPI, this process will hang.\n")
            sys.stderr.write("*****************************************************\n")
            sys.stderr.flush()
            raise e


def make_dir(path):
    """Build directory"""
    if not os.path.exists(path):
        os.makedirs(path)


def get_all_edge_type(etype2files, symmetry):
    """ get_all_edge_type """
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


def get_edge_type(etype, symmetry):
    """ get edge type with etype  """
    if symmetry:
        etype_list = []
        ori_type_list = etype.split(',')
        for i in range(0, len(ori_type_list)):
            etype_list.append(ori_type_list[i])
            r_etype = get_inverse_etype(ori_type_list[i])
            if r_etype != ori_type_list[i]:
                etype_list.append(r_etype)
    else:
        etype_list = etype.split(',')
    return etype_list


def get_sub_path(type2files_dict, sub_types_list, is_edge):
    """ get sub path in metapath split """
    final_path = []
    is_reverse_edge_map = []
    for sub_type in sub_types_list:
        combine = []
        #combine.append(sub_type)
        if sub_type in type2files_dict:
            combine.append(sub_type)
            sub_path = type2files_dict[sub_type]
            combine.append(sub_path)
            if is_edge:
                is_reverse_edge_map.append(0)
        elif is_edge and get_inverse_etype(sub_type) in type2files_dict:
            sub_inverse_type = get_inverse_etype(sub_type)
            combine.append(sub_type)
            sub_path = type2files_dict[sub_inverse_type]
            combine.append(sub_path)
            if is_edge:
                is_reverse_edge_map.append(1)
        combine_path = ":".join(combine)
        if combine_path not in final_path:
            final_path.append(combine_path)
    return ','.join(final_path), is_reverse_edge_map


def change_metapath_index(meta_path, node_type_size, edge_type_size):
    """ change metapath index """
    node_sizes = node_type_size.split(';')
    node_size_dict = {}
    for i in range(0, len(node_sizes)):
        node_type = node_sizes[i].split(":")[0]
        node_size = int(node_sizes[i].split(":")[1])
        node_size_dict[node_type] = node_size
    edge_sizes = edge_type_size.split(';')
    edge_size_dict = {}
    for i in range(0, len(edge_sizes)):
        edge_type = edge_sizes[i].split(":")[0]
        edge_size = int(edge_sizes[i].split(":")[1])
        edge_size_dict[edge_type] = edge_size

    meta_paths = meta_path.split(';')
    meta_path_dict = []
    for i in range(0, len(meta_paths)):
        first_node = meta_paths[i].split('2')[0]
        first_edge = meta_paths[i].split('-')[0]
        meta_path_dict.append({})
        meta_path_dict[i]['path'] = meta_paths[i]
        meta_path_dict[i]['node_size'] = node_size_dict[first_node]
        meta_path_dict[i]['edge_size'] = edge_size_dict[first_edge]
    sort_meta_path = sorted(meta_path_dict, key=lambda x: (x['node_size'], x['edge_size']))
    final_path_list = []
    for i in range(0, len(sort_meta_path)):
        final_path_list.append(sort_meta_path[i]['path'])
    return final_path_list


def get_inverse_etype(etype):
    """ get_inverse_etype """
    fields = etype.split("2")
    if len(fields) == 3:
        src, etype, dst = fields
        r_etype = "2".join([dst, etype, src])
    else:
        r_etype = "2".join([fields[1], fields[0]])
    return r_etype


def get_first_node_type(meta_path):
    """ get first node type from meta path """
    meta_path_vec = []
    if meta_path[0] == '[':
        # multi tensor pair
        assert(meta_path[-1] == ']')
        meta_path_vec.extend(meta_path[1:-2].split(','))
    else:
        meta_path_vec.append(meta_path)
        
    first_node_type_vec = []
    for meta_path in meta_path_vec:
        first_node = []
        meta_paths = meta_path.split(';')
        for i in range(len(meta_paths)):
            tmp_node = meta_paths[i].split('2')[0]
            first_node.append(tmp_node)
        first_node_type_vec.append(";".join(first_node))
    return '[' + ",".join(first_node_type_vec) + ']', len(meta_path_vec)


def parse_files(type_files):
    """ parse_files """
    type2files = OrderedDict()
    for item in type_files.split(","):
        t, file_or_dir = item.split(":")
        type2files[t] = file_or_dir
    return type2files


def get_files(edge_file_or_dir):
    """ get_files """
    if os.path.isdir(edge_file_or_dir):
        ret_files = []
        files = sorted(glob.glob(os.path.join(edge_file_or_dir, "*")))
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
    """ load_ip_addr """
    if isinstance(ip_config, str):
        ip_addr_list = []
        with open(ip_config, 'r') as f:
            for line in f:
                ip_addr_list.append(line.strip())
        ip_addr = ";".join(ip_addr_list)
    elif isinstance(ip_config, list):
        ip_addr = ";".join(ip_config)
    else:
        raise TypeError("ip_config should be list of IP address or "
                        "a path of IP configuration file. "
                        "But got %s" % (type(ip_config)))
    return ip_addr


def convert_nfeat_info(nfeat_info):
    """ convert_nfeat_info """
    res = defaultdict(dict)
    for item in nfeat_info:
        res[item[0]].update({item[1]: [item[2], item[3]]})
    return res


def gzshell_upload(fs_name, fs_user, fs_password, local_path, remote_path):
    """ upload data with gzshell in afs """
    gzshell = os.getenv("GZSHELL")
    client_conf = os.getenv("CLIENT_CONF")
    cmd = "%s --uri=%s --username=%s --password=%s --conf=%s \
            --thread=100 -put %s/ %s" % (gzshell, fs_name, fs_user, \
            fs_password, client_conf, local_path, remote_path)
    log.info("cmd: " + cmd)
    upload_res = run_cmd_get_return_code(cmd)
    retry_num = 0
    while upload_res != 0:
        if retry_num > 3:
            log.info("upload model failed exceeds retry num limit!")
            break
        upload_res = run_cmd_get_return_code(cmd)
        retry_num += 1
    if upload_res != 0:
        log.info("Fail to upload afs model")
        exit(-1)


def gzshell_download(fs_name, fs_user, fs_password, remote_path, local_path):
    """ download data with gzshell in afs """
    gzshell = os.getenv("GZSHELL")
    client_conf = os.getenv("CLIENT_CONF")
    cmd = "%s --uri=%s --username=%s --password=%s --conf=%s \
            --thread=100 -get %s %s" % (gzshell, fs_name, fs_user, \
            fs_password, client_conf, remote_path, local_path)
    log.info("cmd: " + cmd)
    download_res = run_cmd_get_return_code(cmd)
    retry_num = 0
    while download_res != 0:
        if retry_num > 3:
            log.info("download model failed exceeds retry num limit!")
            break
        download_res = run_cmd_get_return_code(cmd)
        retry_num += 1
    if download_res != 0:
        log.info("Fail to download afs model")
        exit(-1)


def run_cmd_get_return_code(cmd):
    """
    run cmd and get its return code, 0 means correct
    """
    return int(core.run_cmd(cmd + "; echo $?").strip().split('\n')[-1])


def run_cmd(cmd):
    """
    run cmd and check result
    """
    ret = run_cmd_get_return_code(cmd)
    if ret != 0:
        raise RuntimeError("Fail to run cmd[%s] ret[%d]" % (cmd, ret))
    return ret


def sample_list_to_str(sage_mode, samples):
    """
    turn sample list config to string
    """
    str_samples = ""
    if sage_mode and samples and len(samples) > 0:
        for s in samples:
            str_samples += str(s)
            str_samples += ";"
        str_samples = str_samples[:-1]
    return str_samples


def print_tensor(scope, name):
    """
    get tensor specified by name
    """
    var = scope.find_var(name)
    if var is None:
        print("var: %s is not in scope" % (name))
        return

    batch_size_tensor = None
    try:
        batch_size_tensor = var.get_tensor()
    except Exception as e:
        print("var: %s is invalid, can not be printed" % (name))
        return

    ori_array = np.array(batch_size_tensor)
    ori_array = ori_array.transpose()
    num = 1
    for e in ori_array.shape:
        num = num * e

    sys.stdout.write("%s:[%s]" % (name, num))
    a = ori_array.reshape(num)

    if num > 10:
        num = 10

    for i in range(num):
        sys.stdout.write(str(a[i]) + ",")
    sys.stdout.write("\n")


def print_tensor_of_program(scope, program):
    """
    print tensor in program
    """
    for param in program.global_block().all_parameters():
        print_tensor(scope, param.name)
