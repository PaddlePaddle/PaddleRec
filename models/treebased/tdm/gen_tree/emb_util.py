# -*- coding=utf8 -*-
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

import os
import paddle
import paddle.fluid as fluid
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    default="create_fake_emb",
    choices=["create_fake_emb", "save_item_emb"],
    type=str,
    help=".")
parser.add_argument("--emb_id_nums", default=13, type=int, help=".")
parser.add_argument("--emb_shape", default=64, type=int, help=".")
parser.add_argument("--emb_path", default='./item_emb.txt', type=str, help='.')
args = parser.parse_args()


def create_fake_emb(emb_id_nums, emb_shape, emb_path):
    x = fluid.data(name="item", shape=[1], lod_level=1, dtype="int64")

    # use layers.embedding to init emb value
    item_emb = fluid.layers.embedding(
        input=x,
        is_sparse=True,
        size=[emb_id_nums, emb_shape],
        param_attr=fluid.ParamAttr(
            name="Item_Emb",
            initializer=fluid.initializer.TruncatedNormal(
                loc=0.0, scale=2.0)))

    # run startup to init emb tensor
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    # get np.array(emb_tensor)
    print("Get Emb")
    item_emb_array = np.array(fluid.global_scope().find_var("Item_Emb")
                              .get_tensor())
    with open(emb_path, 'w+') as f:
        emb_str = ""
        for index, value in enumerate(item_emb_array):
            line = []
            for v in value:
                line.append(str(v))
            line_str = " ".join(line)
            line_str += "\t"
            line_str += str(index)
            line_str += "\n"
            emb_str += line_str
        f.write(emb_str)
    print("Item Emb write Finish")


if __name__ == "__main__":
    create_fake_emb(args.emb_id_nums, args.emb_shape, args.emb_path)
