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

import json
import pickle
import time
import os
import numpy as np

from anytree import (AsciiStyle, LevelOrderGroupIter, LevelOrderIter, Node,
                     NodeMixin, RenderTree)
from anytree.importer.dictimporter import DictImporter
from anytree.iterators.abstractiter import AbstractIter
from anytree.walker import Walker
from tree_impl import TDMTreeClass


class myLevelOrderIter(AbstractIter):
    @staticmethod
    def _iter(children, filter_, stop, maxlevel):
        level = 1
        while children:
            next_children = []
            for child in children:
                if filter_(child):
                    yield child, level
                next_children += AbstractIter._get_children(child.children,
                                                            stop)
            children = next_children
            level += 1
            if AbstractIter._abort_at_level(level, maxlevel):
                break


class Tree_search(object):
    def __init__(self, tree_path, id2item_path, child_num=2):
        self.root = None
        self.id2item = None
        self.item2id = None
        self.child_num = child_num

        self.load(tree_path)
        # self.load_id2item(id2item_path)

        self.level_code = [[]]
        self.max_level = 0
        self.keycode_id_dict = {}
        #  embedding
        self.keycode_nodeid_dict = {}
        self.tree_info = []
        self.id_node_dict = {}

        self.get_keycode_mapping()
        self.travel_tree()
        self.get_children()

    def get_keycode_mapping(self):
        nodeid = 0
        self.embedding = []
        print("Begin Keycode Mapping")
        for node in myLevelOrderIter(self.root):
            node, level = node
            if level - 1 > self.max_level:
                self.max_level = level - 1
                self.level_code.append([])
            if node.ids is not None:
                self.keycode_id_dict[node.key_code] = node.ids
                self.id_node_dict[node.ids] = node
            self.keycode_nodeid_dict[node.key_code] = nodeid
            self.level_code[self.max_level].append(nodeid)

            node_infos = []
            if node.ids is not None:  # item_id
                node_infos.append(node.ids)
            else:
                node_infos.append(0)
            node_infos.append(self.max_level)  # layer_id
            if node.parent:  # ancestor_id
                node_infos.append(self.keycode_nodeid_dict[
                    node.parent.key_code])
            else:
                node_infos.append(0)
            self.tree_info.append(node_infos)
            self.embedding.append(node.emb_vec)
            nodeid += 1
            if nodeid % 1000 == 0:
                print("travel node id {}".format(nodeid))

    def load(self, path):
        print("Begin Load Tree")
        f = open(path, "rb")
        data = pickle.load(f)
        pickle.dump(data, open(path, "wb"), protocol=2)
        importer = DictImporter()
        self.root = importer.import_(data)
        f.close()

    def load_id2item(self, path):
        """load dict from json file"""
        with open(path, "rb") as json_file:
            self.id2item = json.load(json_file)

        self.item2id = {value: int(key) for key, value in self.id2item.items()}

    def get_children(self):
        """get every node children info"""
        print("Begin Keycode Mapping")
        for node in myLevelOrderIter(self.root):
            node, level = node
            node_id = self.keycode_nodeid_dict[node.key_code]
            child_idx = 0
            if node.children:
                for child in node.children:
                    self.tree_info[node_id].append(self.keycode_nodeid_dict[
                        child.key_code])
                    child_idx += 1
            while child_idx < self.child_num:
                self.tree_info[node_id].append(0)
                child_idx += 1
            if node_id % 1000 == 0:
                print("get children  node id {}".format(node_id))

    def travel_tree(self):
        self.travel_list = []
        tree_walker = Walker()
        print("Begin Travel Tree")
        for item in sorted(self.id_node_dict.keys()):
            node = self.id_node_dict[int(item)]
            paths, _, _ = tree_walker.walk(node, self.root)
            paths = list(paths)
            paths.reverse()
            travel = [self.keycode_nodeid_dict[i.key_code] for i in paths]
            while len(travel) < self.max_level:
                travel.append(0)
            self.travel_list.append(travel)


def tree_search_main(tree_path, id2item_path, output_dir, n_clusters=2):
    print("Begin Tree Search")
    t = Tree_search(tree_path, id2item_path, n_clusters)

    # 1. Walk all leaf nodes, get travel path array
    travel_list = np.array(t.travel_list)
    np.save(os.path.join(output_dir, "travel_list.npy"), travel_list)
    with open(os.path.join(output_dir, "travel_list.txt"), 'w') as fout:
        for i, travel in enumerate(t.travel_list):
            travel = map(str, travel)
            fout.write(','.join(travel))
            fout.write("\n")
    print("End Save tree travel")

    # 2. Walk all layer of tree, get layer array
    layer_num = 0
    with open(os.path.join(output_dir, "layer_list.txt"), 'w') as fout:
        for layer in t.level_code:
            # exclude layer 0
            if layer_num == 0:
                layer_num += 1
                continue
            for idx in range(len(layer) - 1):
                fout.write(str(layer[idx]) + ',')
            fout.write(str(layer[-1]) + "\n")
            print("Layer {} has {} node, the first {}, the last {}".format(
                layer_num, len(layer), layer[0], layer[-1]))
            layer_num += 1
    print("End Save tree layer")

    # 3. Walk all node of tree, get tree info
    tree_info = np.array(t.tree_info)
    np.save(os.path.join(output_dir, "tree_info.npy"), tree_info)
    with open(os.path.join(output_dir, "tree_info.txt"), 'w') as fout:
        for i, node_infos in enumerate(t.tree_info):
            node_infos = map(str, node_infos)
            fout.write(','.join(node_infos))
            fout.write("\n")
    print("End Save tree info")

    # 4. save embedding
    embedding = np.array(t.embedding)
    np.save(os.path.join(output_dir, "tree_emb.npy"), embedding)
    with open(os.path.join(output_dir, "tree_embedding.txt"), "w") as fout:
        for i, emb in enumerate(t.embedding):
            emb = map(str, emb)
            fout.write(','.join(emb))
            fout.write("\n")


if __name__ == "__main__":
    tree_path = "./tree.pkl"
    id2item_path = "./id2item.json"
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.system("mkdir -p " + output_dir)
    tree_search_main(tree_path, id2item_path, output_dir)
