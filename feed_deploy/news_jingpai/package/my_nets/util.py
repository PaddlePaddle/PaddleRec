import paddle
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
import os
import numpy as np
import config
from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
from paddle.fluid.incubate.fleet.utils.hdfs import HDFSClient
import collections
import json
import time

fleet_util = FleetUtil()

def print_global_metrics(scope, stat_pos_name, stat_neg_name, sqrerr_name,
                         abserr_name, prob_name, q_name, pos_ins_num_name, 
                         total_ins_num_name, print_prefix):
        auc, bucket_error, mae, rmse, actual_ctr, predicted_ctr, copc,\
            mean_predict_qvalue, total_ins_num = fleet_util.get_global_metrics(\
            scope, stat_pos_name, stat_neg_name, sqrerr_name, abserr_name,\
            prob_name, q_name, pos_ins_num_name, total_ins_num_name)
        log_str = "AUC=%.6f BUCKET_ERROR=%.6f MAE=%.6f " \
                  "RMSE=%.6f Actural_CTR=%.6f Predicted_CTR=%.6f " \
                  "COPC=%.6f MEAN Q_VALUE=%.6f Ins number=%s" % \
                  (auc, bucket_error, mae, rmse, \
                  actual_ctr, predicted_ctr, copc, mean_predict_qvalue, \
                  total_ins_num)
        fleet_util.rank0_print(print_prefix + " " + log_str)
        return print_prefix + " " + log_str #print_prefix + "\n " + log_str

def write_stdout(stdout_str):
    if fleet.worker_index() != 0:
        fleet._role_maker._barrier_worker()
        return
    hadoop_home="$HADOOP_HOME"
    configs = {"fs.default.name": config.fs_name, "hadoop.job.ugi": config.fs_ugi}
    client = HDFSClient(hadoop_home, configs)
    out_dir = config.output_path + "/stdout/"
    if not client.is_exist(out_dir):
        client.makedirs(out_dir)
    job_id_with_host = os.popen("echo -n ${JOB_ID}").read().strip()
    instance_id = os.popen("echo -n ${INSTANCE_ID}").read().strip()
    start_pos = instance_id.find(job_id_with_host)
    end_pos = instance_id.find("--")
    if start_pos != -1 and end_pos != -1:
        job_id_with_host = instance_id[start_pos:end_pos]
    file_path = out_dir + job_id_with_host
    if client.is_file(file_path):
        pre_content = client.cat(file_path)
        with open(job_id_with_host, "w") as f:
            f.write(pre_content + "\n")
            f.write(stdout_str + "\n")
        client.delete(file_path)
        client.upload(out_dir, job_id_with_host, multi_processes=1, overwrite=False)
    else:
        with open(job_id_with_host, "w") as f:
            f.write(stdout_str + "\n")
        client.upload(out_dir, job_id_with_host, multi_processes=1, overwrite=False)
    fleet_util.rank0_info("write %s succeed" % file_path)
    fleet._role_maker._barrier_worker()

def _get_xbox_str(day, model_path, xbox_base_key, data_path, monitor_data, mode="patch"):
    xbox_dict = collections.OrderedDict()
    if mode == "base":
        xbox_dict["id"] = str(xbox_base_key)
    elif mode == "patch":
        xbox_dict["id"] = str(int(time.time()))
    else:
        print("warning: unknown mode %s, set it to patch" % mode)
        mode = "patch"
        xbox_dict["id"] = str(int(time.time()))
    xbox_dict["key"] = str(xbox_base_key)
    if model_path.startswith("hdfs:") or model_path.startswith("afs:"):
        model_path = model_path[model_path.find(":") + 1:]
    xbox_dict["input"] = config.fs_name + model_path.rstrip("/") + "/000"
    xbox_dict["record_count"] = "111111"
    xbox_dict["partition_type"] = "2"
    xbox_dict["job_name"] = "default_job_name"
    xbox_dict["ins_tag"] = "feasign"
    xbox_dict["ins_path"] = data_path
    job_id_with_host = os.popen("echo -n ${JOB_ID}").read().strip()
    instance_id = os.popen("echo -n ${INSTANCE_ID}").read().strip()
    start_pos = instance_id.find(job_id_with_host)
    end_pos = instance_id.find("--")
    if start_pos != -1 and end_pos != -1:
        job_id_with_host = instance_id[start_pos:end_pos]
    xbox_dict["job_id"] = job_id_with_host
    xbox_dict["monitor_data"] = monitor_data
    xbox_dict["monitor_path"] = config.output_path.rstrip("/") + "/monitor/" \
                                + day + ".txt"
    xbox_dict["mpi_size"] = str(fleet.worker_num())
    return json.dumps(xbox_dict)

def write_xbox_donefile(day, pass_id, xbox_base_key, data_path, donefile_name=None, monitor_data=""):
    if fleet.worker_index() != 0:
        fleet._role_maker._barrier_worker()
        return
    day = str(day)
    pass_id = str(pass_id)
    xbox_base_key = int(xbox_base_key)
    mode = None
    if pass_id != "-1":
        mode = "patch"
        suffix_name = "/%s/delta-%s/" % (day, pass_id)
        model_path = config.output_path.rstrip("/") + suffix_name
        if donefile_name is None:
            donefile_name = "xbox_patch_done.txt"
    else:
        mode = "base"
        suffix_name = "/%s/base/" % day
        model_path = config.output_path.rstrip("/") + suffix_name
        if donefile_name is None:
            donefile_name = "xbox_base_done.txt"
    if isinstance(data_path, list):
        data_path = ",".join(data_path)

    if fleet.worker_index() == 0:
        donefile_path = config.output_path + "/" + donefile_name
        xbox_str = _get_xbox_str(day, model_path, xbox_base_key, data_path, monitor_data, mode)
        configs = {"fs.default.name": config.fs_name, "hadoop.job.ugi": config.fs_ugi}
        client = HDFSClient("$HADOOP_HOME", configs)
        if client.is_file(donefile_path):
            pre_content = client.cat(donefile_path)
            last_dict = json.loads(pre_content.split("\n")[-1])
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
                    config.output_path,
                    donefile_name,
                    multi_processes=1,
                    overwrite=False)
                fleet_util.rank0_info("write %s/%s %s succeed" % \
                                       (day, pass_id, donefile_name))
            else:
                fleet_util.rank0_error("not write %s because %s/%s already "
                                       "exists" % (donefile_name, day, pass_id))
        else:
            with open(donefile_name, "w") as f:
                f.write(xbox_str + "\n")
            client.upload(
                config.output_path,
                donefile_name,
                multi_processes=1,
                overwrite=False)
            fleet_util.rank0_error("write %s/%s %s succeed" % \
                                   (day, pass_id, donefile_name))
    fleet._role_maker._barrier_worker()

def jingpai_load_paddle_model(old_startup_program_bin,
                              old_train_program_bin,
                              old_model_path,
                              old_slot_list,
                              new_slot_list,
                              model_all_vars,
                              new_scope,
                              modify_layer_names):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    old_scope = fluid.Scope()
    old_program = fluid.Program()
    old_program = old_program.parse_from_string(open(old_train_program_bin, "rb").read())
    old_startup_program = fluid.Program()
    old_startup_program = old_startup_program.parse_from_string(open(old_startup_program_bin, "rb").read())
    with fluid.scope_guard(old_scope):
        exe.run(old_startup_program)
        variables =  [old_program.global_block().var(i) for i in model_all_vars]
        if os.path.isfile(old_model_path):
            path = os.path.dirname(old_model_path)
            path = "./" if path == "" else path
            filename = os.path.basename(old_model_path)
            fluid.io.load_vars(exe, path, old_program, vars=variables, filename=filename)
        else:
            fluid.io.load_vars(exe, old_model_path, old_program, vars=variables)

    old_pos = {}
    idx = 0
    for i in old_slot_list:
        old_pos[i] = idx
        idx += 1

    for i in modify_layer_names:
        if old_scope.find_var(i) is None:
            print("%s not found in old scope, skip" % i)
            continue
        elif new_scope.find_var(i) is None:
            print("%s not found in new scope, skip" % i)
            continue
        old_param = old_scope.var(i).get_tensor()
        old_param_array =  np.array(old_param).astype("float32")
        old_shape = old_param_array.shape
        #print  i," old_shape ", old_shape

        new_param = new_scope.var(i).get_tensor()
        new_param_array = np.array(new_param).astype("float32")
        new_shape = new_param_array.shape
        #print i," new_shape ", new_shape

        per_dim = len(new_param_array) / len(new_slot_list)
        #print "len(new_param_array) ",len(new_param_array),\
        #  "len(new_slot_list) ", len(new_slot_list)," per_dim ", per_dim

        idx = -per_dim
        for s in new_slot_list:
            idx += per_dim
            if old_pos.get(s) is None:
                    continue                
            for j in range(0, per_dim):
                #print i," row/value ", idx + j, " copy from ", old_pos[s] * per_dim + j
                # a row or a value
                new_param_array[idx + j] = old_param_array[old_pos[s] * per_dim + j]

        new_param.set(new_param_array, place)

    for i in model_all_vars:
        if i in modify_layer_names:
            continue
        old_param = old_scope.find_var(i).get_tensor()
        old_param_array =  np.array(old_param).astype("float32")
        new_param = new_scope.find_var(i).get_tensor()
        new_param.set(old_param_array, place)


def reqi_changeslot(hdfs_dnn_plugin_path, join_save_params, common_save_params, update_save_params, scope2, scope3):
    if fleet.worker_index() != 0:
        return

    print("load paddle model %s" % hdfs_dnn_plugin_path)

    os.system("rm -rf dnn_plugin/ ; hadoop fs -D hadoop.job.ugi=%s -D fs.default.name=%s -get %s ." % (config.fs_ugi, config.fs_name, hdfs_dnn_plugin_path))

    new_join_slot = []
    for line in open("slot/slot", 'r'):
        slot = line.strip()
        new_join_slot.append(slot)
    old_join_slot = []
    for line in open("old_slot/slot", 'r'):
        slot = line.strip()
        old_join_slot.append(slot)

    new_common_slot = []
    for line in open("slot/slot_common", 'r'):
        slot = line.strip()
        new_common_slot.append(slot)
    old_common_slot = []
    for line in open("old_slot/slot_common", 'r'):
        slot = line.strip()
        old_common_slot.append(slot)


    jingpai_load_paddle_model("old_program/old_join_common_startup_program.bin",
                              "old_program/old_join_common_train_program.bin",
                              "dnn_plugin/paddle_dense.model.0",
                              old_join_slot,
                              new_join_slot,
                              join_save_params,
                              scope2,
                              ["join.batch_size","join.batch_sum","join.batch_square_sum","join_0.w_0"])

    jingpai_load_paddle_model("old_program/old_join_common_startup_program.bin",
                              "old_program/old_join_common_train_program.bin",
                              "dnn_plugin/paddle_dense.model.1",
                              old_common_slot,
                              new_common_slot,
                              common_save_params,
                              scope2,
                              ["common.batch_size","common.batch_sum","common.batch_square_sum","common_0.w_0"])

    jingpai_load_paddle_model("old_program/old_update_startup_program.bin",
                              "old_program/old_update_main_program.bin",
                              "dnn_plugin/paddle_dense.model.2",
                              old_join_slot,
                              new_join_slot,
                              update_save_params,
                              scope3,
                              ["fc_0.w_0"])
