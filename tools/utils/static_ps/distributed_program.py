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
"""Distributed Program
"""
import copy

import paddle
import paddle.static as static

from place import get_cuda_places


def set_dump_config(args, model):
    """ set demp config """
    if not getattr(args, "need_dump", False):
        return
    
    program = model.train_program
    opt_info = program._fleet_opt
    # add default dump path
    dump_fields_path = getattr(args, "dump_fields_path", None)
    if dump_fields_path is None:
        dump_fields_path =  "/raid0/dump_log/"
    print("dump_fields_path=", dump_fields_path)
    opt_info["dump_fields_path"] = dump_fields_path
    opt_info["dump_fields_mode"] = "a"
    opt_info["dump_param"] = []
    opt_info["dump_fields"] = []
    
    all_param_list = []
    for param in program.all_parameters():
        if param.type != core.VarDesc.VarType.DENSE_TENSOR:
            continue
        all_param_list.append(param.name)
        all_param_list.append(param.name + '@GRAD')
        
    all_field_list = []
    for var in program.list_vars():
        if var.type != core.VarDesc.VarType.DENSE_TENSOR:
            continue
        if var.persistable:
            continue
        if var.name in all_param_list:
            continue
        all_field_list.append(var.name)
        
    dump_params = getattr(args, "dump_param", [])
    if len(dump_params) > 0:
        for name in dump_params:
            if not name in all_param_list:
                continue
            opt_info["dump_param"].append(name)
    
    dump_fields = getattr(args, "dump_fields", [])
    if len(dump_fields) > 0:
        for name in dump_fields:
            new_name = name
            pos = name.find("@")
            if pos > 0:
                new_name = name[0: pos]
            if not new_name in all_field_list:
                continue
            opt_info["dump_fields"].append(name)
    
    print("dump_field[%s]: %s" % (len(opt_info["dump_fields"]), str(opt_info["dump_fields"])))
    print("dump_param[%s]: %s" % (len(opt_info["dump_param"]), str(opt_info["dump_param"])))
    
    model.train_program._fleet_opt = opt_info
    print(str(opt_info))

def make_distributed_train_program(args, model_dict):
    """doc"""
    device_ids = get_cuda_places()
    train_opt = copy.deepcopy(static.default_main_program()._fleet_opt)

    #print("train opt = ", train_opt)

    model_dict.startup_program = static.default_startup_program()
    model_dict.train_program = static.default_main_program().clone()
    model_dict.train_program._fleet_opt = train_opt
    model_dict.train_program._fleet_opt['worker_places'] = device_ids
    # add dump config
    set_dump_config(args, model_dict)

    with open("join_main_program.pbtxt", "w") as fout:
        fout.write(str(model_dict.train_program))
    with open("join_startup_program.pbtxt", "w") as fout:
        fout.write(str(model_dict.startup_program))


def make_distributed_infer_program(args, model_dict):
    """doc"""
    device_ids = get_cuda_places()
    infer_opt = copy.deepcopy(static.default_main_program()._fleet_opt)
    model_dict.train_program = static.default_main_program().clone()
    model_dict.train_program._fleet_opt = infer_opt
    opt_info = model_dict.train_program._fleet_opt

    opt_info['worker_places'] = device_ids
    opt_info["dump_fields"] = [
        args.dump_node_name + ".tmp_0", args.dump_node_emb_name + ".tmp_0"
    ]
    opt_info["dump_fields_path"] = args.local_result_path
    opt_info["is_dump_in_simple_mode"] = True
    opt_info["user_define_dump_filename"] = "000" 
    opt_info["dump_fields_mode"] = "a"
    opt_info["dump_num_decimals"] = 9
    opt_info["use_ps_gpu"] = True
    opt_info["use_gpu_graph"] = True

    with open("infer_before_main_program.pbtxt", "w") as fout:
        fout.write(str(model_dict.train_program))
    remove_op(model_dict.train_program, "push_gpups_sparse")
    remove_backword(model_dict.train_program)
    with open("infer_main_program.pbtxt", "w") as fout:
        fout.write(str(model_dict.train_program))
    
    # infer remove comm init
    remove_op(model_dict.startup_program, "c_comm_init_all")
    remove_op(model_dict.startup_program, "c_gen_nccl_id")
    remove_op(model_dict.startup_program, "c_comm_init_multitrainer")
    with open("infer_startup_program.pbtxt", "w") as fout:
        fout.write(str(model_dict.startup_program))


def remove_op(program, name):
    """
    remove op
    """
    block = program.global_block()
    for ids, op in list(enumerate(block.ops)):
        if op.type == name:
            block._remove_op(ids)
            return


def remove_backword(program):
    """
    remove_backword
    """
    block = program.global_block()
    last_idx = -1
    for ids, op in list(enumerate(block.ops)):
        if op.has_attr("is_test"):
            op._set_attr("is_test", True)
    for ids, op in list(enumerate(block.ops)):
        if op._is_backward_op():
            last_idx = ids
            break
    last_idx -= 1  # remove fill_constant
    for ids, op in list(enumerate(block.ops)):
        if ids > last_idx:
            block._remove_op(last_idx + 1)
