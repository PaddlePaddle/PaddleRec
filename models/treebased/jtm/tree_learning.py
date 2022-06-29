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
from paddle.distributed.fleet.dataset import TreeIndex
import numpy as np
import random
import os
import sys
import multiprocessing as mp
import json
import time
import math
import argparse
from user_preference import UserPreferenceModel

paddle.enable_static()


def mp_run(data, process_num, func, *args):
    """ run func with multi process
    """
    level_start = time.time()
    partn = max(len(data) / process_num, 1)
    start = 0
    p_idx = 0
    ps = []
    while start < len(data):
        local_data = data[start:start + partn]
        start += partn
        p = mp.Process(target=func, args=(local_data, p_idx) + args)
        ps.append(p)
        p.start()
        p_idx += 1
    for p in ps:
        p.join()

    for p in ps:
        p.terminate()
    return p_idx


def get_itemset_given_ancestor(pi_new, node):
    res = []
    for ci, code in pi_new.items():
        if code == node:
            res.append(ci)
    return res


# you need to define your sample_set
def get_sample_set(ck, args):
    if not os.path.exists("{}/samples_{}.json".format(args.sample_directory,
                                                      ck)):
        return []
    with open("{}/samples_{}.json".format(args.sample_directory, ck),
              'r') as f:
        all_samples = json.load(f)

    sample_nums = args.sample_nums
    if sample_nums > 0:
        size = len(all_samples)
        if (size > sample_nums):
            sample_set = np.random.choice(
                range(size), size=sample_nums, replace=False).tolist()
            return [all_samples[s] for s in sample_set]
    else:
        return all_samples


def get_weights(C_ni, idx, edge_weights, ni, children_of_ni_in_level_l, tree,
                args):
    """use the user preference prediction model to calculate the required weights

    Returns:
        all weights

    Args:
        C_ni (item, required): item set whose ancestor is the non-leaf node ni
        ni (node, required): a non-leaf node in level l-d
        children_of_ni_in_level_l (list, required): the level l-th children of ni
        tree (tree, required): the old tree (\pi_{old})

    """
    #print("begin idx: {}, C_ni: {}.".format(idx, len(C_ni)))
    tree_emb_size = tree.emb_size()
    #print("tree_emb_size: ", tree_emb_size)
    prediction_model = UserPreferenceModel(args.init_model_path, tree_emb_size,
                                           args.node_emb_size)

    for ck in C_ni:
        _weights = list()
        # the first element is the list of nodes in level l
        _weights.append([])
        # the second element is the list of corresponding weights
        _weights.append([])

        samples = get_sample_set(ck, args)
        print(samples)
        for node in children_of_ni_in_level_l:
            path_to_ni = tree.get_travel_path(node, ni)
            if len(samples) == 0:
                weight = 0.0
            else:
                weight = prediction_model.calc_prediction_weight(samples,
                                                                 path_to_ni)

            _weights[0].append(node)
            _weights[1].append(weight)
        edge_weights.update({ck: _weights})


# print("end idx: {}, C_ni: {}, edge_weights: {}.".format(idx, len(C_ni), len(edge_weights)))


def assign_parent(tree, l_max, l, d, ni, C_ni, args):
    """implementation of line 5 of Algorithm 2

    Returns: 
        updated \pi_{new}

    Args:
        l_max (int, required): the max level of the tree
        l (int, required): current assign level
        d (int, required): level gap in tree_learning
        ni (node, required): a non-leaf node in level l-d
        C_ni (item, required): item set whose ancestor is the non-leaf node ni
        tree (tree, required): the old tree (\pi_{old})
    """
    # get the children of ni in level l
    children_of_ni_in_level_l = tree.get_children_codes(ni, l)

    print(children_of_ni_in_level_l)
    # get all the required weights
    edge_weights = mp.Manager().dict()

    mp_run(C_ni, 12, get_weights, edge_weights, ni, children_of_ni_in_level_l,
           tree, args)

    print("finish calculate edge_weights. {}.".format(len(edge_weights)))
    # assign each item to the level l node with the maximum weight
    assign_dict = dict()
    for ci, info in edge_weights.items():
        assign_candidate_nodes = np.array(info[0], dtype=np.int64)
        assign_weights = np.array(info[1], dtype=np.float32)
        sorted_idx = np.argsort(-assign_weights)
        sorted_weights = assign_weights[sorted_idx]
        sorted_candidate_nodes = assign_candidate_nodes[sorted_idx]
        # assign item ci to the node with the largest weight
        max_weight_node = sorted_candidate_nodes[0]
        if max_weight_node in assign_dict:
            assign_dict[max_weight_node].append(
                (ci, 0, sorted_candidate_nodes, sorted_weights))
        else:
            assign_dict[max_weight_node] = [
                (ci, 0, sorted_candidate_nodes, sorted_weights)
            ]

    edge_weights = None

    # get each item's original assignment of level l in tree, used in rebalance process
    origin_relation = tree.get_pi_relation(C_ni, l)
    # for ci in C_ni:
    #     origin_relation[ci] = self._tree.get_ancestor(ci, l)

    # rebalance
    max_assign_num = int(math.pow(2, l_max - l))
    processed_set = set()

    while True:
        max_assign_cnt = 0
        max_assign_node = None

        for node in children_of_ni_in_level_l:
            if node in processed_set:
                continue
            if node not in assign_dict:
                continue
            if len(assign_dict[node]) > max_assign_cnt:
                max_assign_cnt = len(assign_dict[node])
                max_assign_node = node

        if max_assign_node == None or max_assign_cnt <= max_assign_num:
            break

        # rebalance
        processed_set.add(max_assign_node)
        elements = assign_dict[max_assign_node]
        elements.sort(
            key=lambda x: (int(max_assign_node != origin_relation[x[0]]), -x[3][x[1]])
        )
        for e in elements[max_assign_num:]:
            idx = e[1] + 1
            while idx < len(e[2]):
                other_parent_node = e[2][idx]
                if other_parent_node in processed_set:
                    idx += 1
                    continue
                if other_parent_node not in assign_dict:
                    assign_dict[other_parent_node] = [(e[0], idx, e[2], e[3])]
                else:
                    assign_dict[other_parent_node].append(
                        (e[0], idx, e[2], e[3]))
                break

        del elements[max_assign_num:]

    pi_new = dict()
    for parent_code, value in assign_dict.items():
        max_assign_num = int(math.pow(2, l_max - l))
        assert len(value) <= max_assign_num
        for e in value:
            assert e[0] not in pi_new
            pi_new[e[0]] = parent_code

    return pi_new


def process(nodes, idx, pi_new_final, tree, l, d, args):
    l_max = tree.height() - 1
    for ni in nodes:
        C_ni = get_itemset_given_ancestor(pi_new_final, ni)
        print("begin to handle {}, have {} items.".format(ni, len(C_ni)))
        if len(C_ni) == 0:
            continue
        pi_star = assign_parent(tree, l_max, l, d, ni, C_ni, args)
        print(pi_star)
        # update pi_new according to the found optimal pi_star
        for item, node in pi_star.items():
            pi_new_final.update({item: node})
        print("end to handle {}.".format(ni))


def tree_learning(args):
    tree = TreeIndex(args.tree_name, args.tree_path)
    d = args.gap

    l_max = tree.height() - 1
    l = d

    pi_new = dict()

    all_items = [node.id() for node in tree.get_all_leafs()]
    pi_new = tree.get_pi_relation(all_items, l - d)

    pi_new_final = mp.Manager().dict()
    pi_new_final.update(pi_new)

    del all_items
    del pi_new

    while d > 0:
        print("begin to re-assign {} layer by {} layer.".format(l, l - d))
        nodes = tree.get_layer_codes(l - d)
        real_process_num = mp_run(nodes, 12, process, pi_new_final, tree, l, d,
                                  args)
        d = min(d, l_max - l)
        l = l + d
    print(pi_new_final)


if __name__ == '__main__':
    _PARSER = argparse.ArgumentParser(description="Tree Learning Algorith.")
    _PARSER.add_argument("--tree_name", required=True, help="tree name.")
    _PARSER.add_argument("--tree_path", required=True, help="tree path.")
    _PARSER.add_argument(
        "--sample_directory", required=True, help="samples directory")
    _PARSER.add_argument(
        "--output_filename", default="./output.pb", help="new tree filename.")
    _PARSER.add_argument("--gap", type=int, default=7, help="gap.")
    _PARSER.add_argument(
        "--node_emb_size", type=int, default=64, help="node embedding size.")
    _PARSER.add_argument(
        "--sample_nums",
        type=int,
        default=-1,
        help="sample nums. default value is -1, means use all related train samples."
    )
    _PARSER.add_argument(
        "--init_model_path", type=str, default="", help="model path.")
    args = _PARSER.parse_args()

    tree_learning(args)
