# Copyright (C) 2016-2018 Alibaba Group Holding Limited
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

from anytree import NodeMixin, RenderTree
import numpy as np
from anytree.exporter.dictexporter import DictExporter
import pickle
import json
import os
import time


class BaseClass(object):
    pass


class TDMTreeClass(BaseClass, NodeMixin):
    def __init__(self,
                 key_code,
                 emb_vec,
                 ids=None,
                 text=None,
                 parent=None,
                 children=None):
        super(TDMTreeClass, self).__init__()
        self.key_code = key_code
        self.ids = ids
        self.emb_vec = emb_vec
        self.text = text
        self.parent = parent
        if children:
            self.children = children

    def set_parent(self, parent):
        self.parent = parent

    def set_children(self, children):
        self.children = children


def _build(ids, codes, data, items, output_dir, n_clusters=2):
    code_list = [0] * 50000000
    node_dict = {}
    max_code = 0
    id2item = {}
    curtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print('%s start gen code_list' % curtime)
    for _id, code, datum, item in zip(ids, codes, data, items):
        code_list[code] = [datum, _id]
        id2item[str(_id)] = item
        max_code = max(code, max_code)
        ancessors = _ancessors(code, n_clusters)
        for ancessor in ancessors:
            code_list[ancessor] = [[]]

    for code in range(max_code, -1, -1):
        if code_list[code] == 0:
            continue
        if len(code_list[code]) > 1:
            pass
        elif len(code_list[code]) == 1:
            code_list[code][0] = np.mean(code_list[code][0], axis=0)
        if code > 0:
            ancessor = int((code - 1) / n_clusters)
            code_list[ancessor][0].append(code_list[code][0])

    print('start gen node_dict')
    for code in range(0, max_code + 1):
        if code_list[code] == 0:
            continue
        if len(code_list[code]) > 1:
            [datum, _id] = code_list[code]
            node_dict[code] = TDMTreeClass(code, emb_vec=datum, ids=_id)
        elif len(code_list[code]) == 1:
            [datum] = code_list[code]
            node_dict[code] = TDMTreeClass(code, emb_vec=datum)
        if code > 0:
            ancessor = int((code - 1) / n_clusters)
            node_dict[code].set_parent(node_dict[ancessor])

    save_tree(node_dict[0], os.path.join(output_dir, 'tree.pkl'))
    save_dict(id2item, os.path.join(output_dir, 'id2item.json'))


def render(root):
    for row in RenderTree(root, childiter=reversed):
        print("%s%s" % (row.pre, row.node.text))


def save_tree(root, path):
    print('save tree to %s' % path)
    exporter = DictExporter()
    data = exporter.export(root)
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()


def save_dict(dic, filename):
    """save dict into json file"""
    print('save dict to %s' % filename)
    with open(filename, "w") as json_file:
        json.dump(dic, json_file, ensure_ascii=False)


def _ancessors(code, n_clusters):
    ancs = []
    while code > 0:
        code = int((code - 1) / n_clusters)
        ancs.append(code)
    return ancs
