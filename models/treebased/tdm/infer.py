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

from model import dnn_model_define
import paddle
import paddle.fluid as fluid
import os
import time
import numpy as np
import multiprocessing as mp
import sys
from paddle.distributed.fleet.dataset import TreeIndex

paddle.enable_static()


class Reader():
    def __init__(self, item_nums):
        self.item_nums = item_nums

    def line_process(self, line):
        history_ids = [0] * (self.item_nums)
        features = line.strip().split("\t")
        groundtruth = [int(ff) for ff in features[1].split(',')]
        for item in features[2:]:
            slot, feasign = item.split(":")
            slot_id = int(slot.split("_")[1])
            history_ids[slot_id - 1] = int(feasign)

        return groundtruth, history_ids

    def dataloader(self, file_list):
        "DataLoader Pyreader Generator"

        def reader():
            for file in file_list:
                with open(file, 'r') as f:
                    for line in f:
                        groudtruth, output_list = self.line_process(line)
                        yield groudtruth, output_list

        return reader


def net_input(item_nums=69):
    user_input = [
        paddle.static.data(
            name="item_" + str(i + 1), shape=[None, 1], dtype="int64")
        for i in range(item_nums)
    ]

    item = paddle.static.data(name="unit_id", shape=[None, 1], dtype="int64")

    return user_input + [item]


def mp_run(data, process_num, func, *args):
    """ run func with multi process
    """
    level_start = time.time()
    partn = int(max(len(data) / process_num, 1))
    start = 0
    p_idx = 0
    ps = []
    manager = mp.Manager()
    res = manager.dict()
    while start < len(data):
        local_data = data[start:start + partn]
        start += partn
        p = mp.Process(target=func, args=(res, local_data, p_idx) + args)
        ps.append(p)
        p.start()
        p_idx += 1
    for p in ps:
        p.join()

    for p in ps:
        p.terminate()

    total_precision_rate = 0.0
    total_recall_rate = 0.0
    total_nums = 0
    for i in range(p_idx):
        print(i)
        total_recall_rate += res["{}_recall".format(i)]
        total_precision_rate += res["{}_precision".format(i)]
        total_nums += res["{}_nums".format(i)]
    print("global recall rate: {} / {} = {}".format(
        total_recall_rate, total_nums, total_recall_rate / float(total_nums)))
    print("global precision rate: {} / {} = {}".format(
        total_precision_rate, total_nums, total_precision_rate / float(
            total_nums)))

    return p_idx


def load_tree_info(name, path, topk=200):
    tree = TreeIndex(name, path)
    all_codes = []
    first_layer_code = None
    for i in range(tree.height()):
        layer_codes = tree.get_layer_codes(i)
        if len(layer_codes) > topk and first_layer_code == None:
            first_layer_code = layer_codes
        all_codes += layer_codes
    all_ids = tree.get_nodes(all_codes)
    id_code_map = {}
    code_id_map = {}
    for i in range(len(all_codes)):
        id = all_ids[i].id()
        code = all_codes[i]
        id_code_map[id] = code
        code_id_map[code] = id
    print(len(all_codes), len(all_ids), len(id_code_map), len(code_id_map))

    first_layer = tree.get_nodes(first_layer_code)
    first_layer = [node.id() for node in first_layer]

    return id_code_map, code_id_map, tree.branch(), first_layer


def infer(res_dict, filelist, process_idx, init_model_path, id_code_map,
          code_id_map, branch, first_layer_set, config):
    print(process_idx, filelist, init_model_path)
    item_nums = config.get("hyper_parameters.item_nums", 69)
    topk = config.get("hyper_parameters.topk", 200)
    node_nums = config.get("hyper_parameters.sparse_feature_num")
    node_emb_size = config.get("hyper_parameters.node_emb_size")
    input = net_input(item_nums)

    embedding = paddle.nn.Embedding(
        node_nums,
        node_emb_size,
        sparse=True,
        weight_attr=paddle.framework.ParamAttr(
            name="tdm.bw_emb.weight",
            initializer=paddle.nn.initializer.Normal(std=0.001)))

    user_feature = input[0:item_nums]
    user_feature_emb = list(map(embedding, user_feature))  # [(bs, emb)]

    unit_id_emb = embedding(input[-1])
    dout = dnn_model_define(user_feature_emb, unit_id_emb)

    softmax_prob = paddle.nn.functional.softmax(dout)
    positive_prob = paddle.slice(softmax_prob, axes=[1], starts=[1], ends=[2])
    prob_re = paddle.reshape(positive_prob, [-1])

    _, topk_i = paddle.topk(prob_re, k=topk)
    topk_node = paddle.index_select(input[-1], topk_i)

    with open("main_program", 'w') as f:
        f.write(str(paddle.static.default_main_program()))

    exe = paddle.static.Executor(fluid.CPUPlace())
    exe.run(paddle.static.default_startup_program())

    print("begin to load parameters")
    #fluid.io.load_persistables(exe, dirname=init_model_path)
    paddle.static.load(paddle.static.default_main_program(),
                       init_model_path + '/rec_static')
    print("end load parameters")
    reader_instance = Reader(item_nums)
    reader = reader_instance.dataloader(filelist)

    total_recall_rate = 0.0
    total_precision_rate = 0.0
    total_nums = 0
    child_info = dict()
    for groudtruth, user_input in reader():
        total_nums += 1

        recall_result = []
        candidate = first_layer_set

        idx = 8
        while (len(recall_result) < topk):
            idx += 1
            feed_dict = {}
            for i in range(1, 70):
                feed_dict['item_' + str(i)] = np.ones(
                    shape=[len(candidate), 1],
                    dtype='int64') * user_input[i - 1]
            feed_dict['unit_id'] = np.array(
                candidate, dtype='int64').reshape(-1, 1)

            res = exe.run(program=paddle.static.default_main_program(),
                          feed=feed_dict,
                          fetch_list=[topk_node.name])
            topk_node_res = res[0].reshape([-1]).tolist()

            candidate = []
            for i in range(len(topk_node_res)):
                node = topk_node_res[i]
                if node not in child_info:
                    child_info[node] = []
                    node_code = id_code_map[node]
                    for j in range(1, branch + 1):
                        child_code = node_code * branch + j
                        if child_code in code_id_map:
                            child_info[node].append(code_id_map[child_code])

                if len(child_info[node]) == 0:
                    recall_result.append(node)
                else:
                    candidate = candidate + child_info[node]

        recall_result = recall_result[:topk]
        intersec = list(set(recall_result).intersection(set(groudtruth)))
        total_recall_rate += float(len(intersec)) / float(len(groudtruth))
        total_precision_rate += float(len(intersec)) / float(
            len(recall_result))

        if (total_nums % 100 == 0):
            print("global recall rate: {} / {} = {}".format(
                total_recall_rate, total_nums, total_recall_rate / float(
                    total_nums)))
            print("global precision rate: {} / {} = {}".format(
                total_precision_rate, total_nums, total_precision_rate / float(
                    total_nums)))
    res_dict["{}_recall".format(process_idx)] = total_recall_rate
    res_dict["{}_precision".format(process_idx)] = total_precision_rate
    res_dict["{}_nums".format(process_idx)] = total_nums
    print("process idx:{}, global recall rate: {} / {} = {}".format(
        process_idx, total_recall_rate, total_nums, total_recall_rate / float(
            total_nums)))
    print("process idx:{}, global precision rate: {} / {} = {}".format(
        process_idx, total_precision_rate, total_nums, total_precision_rate /
        float(total_nums)))


if __name__ == '__main__':
    utils_path = "{}/tools/utils/static_ps".format(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
    sys.path.append(utils_path)
    print(utils_path)
    import common
    yaml_helper = common.YamlHelper()
    config = yaml_helper.load_yaml(sys.argv[1])

    test_files_path = "../demo_data/test_data"
    filelist = [
        "{}/{}".format(test_files_path, x) for x in os.listdir(test_files_path)
    ]
    print(filelist)
    init_model_path = sys.argv[2]
    print(init_model_path)
    tree_name = config.get("hyper_parameters.tree_name")
    tree_path = config.get("hyper_parameters.tree_path")
    print("tree_name: {}".format(tree_name))
    print("tree_path: {}".format(tree_path))
    id_code_map, code_id_map, branch, first_layer_set = load_tree_info(
        tree_name, tree_path)
    mp_run(filelist, 12, infer, init_model_path, id_code_map, code_id_map,
           branch, first_layer_set, config)
