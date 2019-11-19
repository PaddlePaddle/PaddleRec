import paddle
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
import os
import numpy as np
import config

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
