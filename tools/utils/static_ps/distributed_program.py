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


def make_distributed_train_program(args, model_dict):
    """doc"""
    device_ids = get_cuda_places()
    train_opt = copy.deepcopy(static.default_main_program()._fleet_opt)

    #print("train opt = ", train_opt)

    model_dict.startup_program = static.default_startup_program()
    model_dict.train_program = static.default_main_program().clone()
    model_dict.train_program._fleet_opt = train_opt
    model_dict.train_program._fleet_opt['worker_places'] = device_ids

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
    #opt_info["dump_file_num"] = 64
    with open("infer_before_main_program.pbtxt", "w") as fout:
        fout.write(str(model_dict.train_program))
    remove_op(model_dict.train_program, "push_gpups_sparse")
    remove_backword(model_dict.train_program)
    with open("infer_main_program.pbtxt", "w") as fout:
        fout.write(str(model_dict.train_program))


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
