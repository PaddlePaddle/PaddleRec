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
import sys
import os
import json
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "-type", type=str, default="train", help="train|valid|test")
parser.add_argument("-maxlen", type=int, default=20)


def load_graph(source):
    graph = {}
    with open(source) as fr:
        for line in fr:
            conts = line.strip().split(',')
            user_id = int(conts[0])
            item_id = int(conts[1])
            time_stamp = int(conts[2])
            if user_id not in graph:
                graph[user_id] = []
            graph[user_id].append((item_id, time_stamp))

    for user_id, value in graph.items():
        value.sort(key=lambda x: x[1])
        graph[user_id] = [x[0] for x in value]
    return graph


if __name__ == "__main__":
    args = parser.parse_args()
    filelist = []
    for i in range(10):
        filelist.append(open(args.type + "/part-%d" % (i), "w"))
    action_graph = load_graph("book_data/book_" + args.type + ".txt")
    if args.type == "train":
        for uid, item_list in action_graph.items():
            for i in range(4, len(item_list)):
                if i >= args.maxlen:
                    hist_item = item_list[i - args.maxlen:i]
                else:
                    hist_item = item_list[:i]
                target_item = item_list[i]
                print(
                    " ".join(["user_id:" + str(uid)] + [
                        "hist_item:" + str(n) for n in hist_item
                    ] + ["target_item:" + str(target_item)]),
                    file=random.choice(filelist))
    else:
        for uid, item_list in action_graph.items():
            k = int(len(item_list) * 0.8)
            if k >= args.maxlen:
                hist_item = item_list[k - args.maxlen:k]
            else:
                hist_item = item_list[:k]
            target_item = item_list[k:]
            print(
                " ".join(["user_id:" + str(uid), "target_item:0"] + [
                    "hist_item:" + str(n) for n in hist_item
                ] + ["eval_item:" + str(n) for n in target_item]),
                file=random.choice(filelist))
