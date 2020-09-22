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

from __future__ import print_function
import numpy as np

import sys
import os
import codecs
from tree_impl import _build

_CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CUR_DIR, ".."))


class TreeBuilder:
    def __init__(self, output_dir='./', n_clusters=2):
        self.output_dir = output_dir
        self.n_clusters = n_clusters

    def build(
            self,
            ids,
            codes,
            data=None,
            items=None,
            id_offset=None, ):
        _build(ids, codes, data, items, self.output_dir, self.n_clusters)

    def _ancessors(self, code):
        ancs = []
        while code > 0:
            code = int((code - 1) / 2)
            ancs.append(code)
        return ancs
