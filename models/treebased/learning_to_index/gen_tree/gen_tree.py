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
import argparse
import time
import argparse
from k_means import Cluster
from tree_search import tree_search_main

parser = argparse.ArgumentParser()
parser.add_argument("--emd_path", default='', type=str, help=".")
parser.add_argument("--emb_size", default=64, type=int, help=".")
parser.add_argument("--thread_num", default=8, type=int, help=".")
parser.add_argument("--n_clusters", default=2, type=int, help=".")
parser.add_argument("--output_dir", default='', type=str, help='.')
args = parser.parse_args()


def main():
    cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    if not os.path.exists(args.output_dir):
        os.system("mkdir -p " + args.output_dir)
    print('%s start build tree' % cur_time)

    # 1. Tree clustering, generating two files in current directory, tree.pkl, id2item.json
    cluster_start_time = time.time()
    cluster = Cluster(
        args.emd_path,
        args.emb_size,
        parall=args.thread_num,
        output_dir=args.output_dir,
        _n_clusters=args.n_clusters)
    cluster.train()
    cluster_end_time = time.time()
    print("Cluster using time: {}".format(
        cluster_end_time - cluster_start_time))

    # 2. Tree searching, generating tree_info, travel_list, layer_list for train process.
    tree_search_start_time = time.time()
    tree_search_main(
        os.path.join(args.output_dir, "tree.pkl"),
        os.path.join(args.output_dir, "id2item.json"), args.output_dir,
        args.n_clusters)
    tree_search_end_time = time.time()
    print("Generate tree meta file using time: {}".format(
        tree_search_end_time - tree_search_start_time))


if __name__ == "__main__":
    main()
