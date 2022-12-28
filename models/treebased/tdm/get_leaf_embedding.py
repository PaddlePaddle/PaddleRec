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

import paddle
import numpy as np
import io
import sys
from paddle.distributed.fleet.dataset import TreeIndex
import os

paddle.enable_static()


def get_emb_numpy(tree_node_num, node_emb_size, init_model_path=""):
    all_nodes = paddle.static.data(
        name="all_nodes",
        shape=[-1, 1],
        dtype="int64",
        lod_level=1, )

    output = paddle.static.nn.embedding(
        input=all_nodes,
        is_sparse=True,
        size=[tree_node_num, node_emb_size],
        param_attr=paddle.ParamAttr(
            name="tdm.bw_emb.weight",
            initializer=paddle.nn.initializer.Uniform()))

    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    exe.run(paddle.static.default_startup_program())
    if init_model_path != "":
        paddle.static.load(paddle.static.default_main_program(),
                           init_model_path + '/rec_static')

    return np.array(paddle.static.global_scope().find_var("tdm.bw_emb.weight")
                    .get_tensor())


if __name__ == '__main__':
    utils_path = "{}/tools/utils/static_ps".format(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
    sys.path.append(utils_path)
    print(utils_path)
    import common

    yaml_helper = common.YamlHelper()
    config = yaml_helper.load_yaml(sys.argv[1])

    tree_name = config.get("hyper_parameters.tree_name")
    tree_path = config.get("hyper_parameters.tree_path")
    tree_node_num = config.get("hyper_parameters.sparse_feature_num")
    node_emb_size = config.get("hyper_parameters.node_emb_size")

    tensor = get_emb_numpy(tree_node_num, node_emb_size, sys.argv[2])

    tree = TreeIndex(tree_name, tree_path)
    all_leafs = tree.get_all_leafs()

    with open(sys.argv[3], 'w') as fout:
        for node in all_leafs:
            node_id = node.id()
            emb_vec = list(map(str, tensor[node_id].tolist()))
            emb_vec = [str(node_id)] + emb_vec
            fout.write(",".join(emb_vec))
            fout.write("\n")
