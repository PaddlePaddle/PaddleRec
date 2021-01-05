#encoding=utf-8
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
import sys
import jieba
import numpy as np
import random

f = open("./raw_data.txt", "r", encoding="utf-8")
lines = f.readlines()
f.close()

#建立字典
word_dict = {}
for line in lines:
    line = line.strip().split("\t")
    text = line[0].strip("") + " " + line[1].strip("")
    text = jieba.cut(text)
    for word in text:
        if word in word_dict:
            continue
        else:
            word_dict[word] = len(word_dict) + 1

f = open("./raw_data.txt", "r", encoding="utf-8")
lines = f.readlines()
f.close()

lines = [line.strip().split("\t") for line in lines]

#建立以query为key，以负例为value的字典
neg_dict = {}
for line in lines:
    if line[2] == "0":
        if line[0] in neg_dict:
            neg_dict[line[0]].append(line[1])
        else:
            neg_dict[line[0]] = [line[1]]

#建立以query为key，以正例为value的字典
pos_dict = {}
for line in lines:
    if line[2] == "1":
        if line[0] in pos_dict:
            pos_dict[line[0]].append(line[1])
        else:
            pos_dict[line[0]] = [line[1]]

print("build dict done")
#划分训练集和测试集
query_list = list(pos_dict.keys())
#print(len(query_list))
np.random.seed(107)
np.random.shuffle(query_list)
train_query = query_list[:11600]
test_query = query_list[11600:]

#获得训练集
train_set = []
for query in train_query:
    for pos in pos_dict[query]:
        if query not in neg_dict:
            continue
        for neg in neg_dict[query]:
            train_set.append([query, pos, neg])
random.shuffle(train_set)
print("get train_set done")

#获得测试集
test_set = []
for query in test_query:
    for pos in pos_dict[query]:
        test_set.append([query, pos, 1])
    if query not in neg_dict:
        continue
    for neg in neg_dict[query]:
        test_set.append([query, neg, 0])
random.shuffle(test_set)
print("get test_set done")

#训练集中的query,pos,neg转化为词袋
f = open("train.txt", "w", encoding="utf-8")
for line in train_set:
    query = jieba.cut(line[0].strip())
    pos = jieba.cut(line[1].strip())
    neg = jieba.cut(line[2].strip())
    query_token = [0] * (len(word_dict) + 1)
    for word in query:
        query_token[word_dict[word]] = 1
    pos_token = [0] * (len(word_dict) + 1)
    for word in pos:
        pos_token[word_dict[word]] = 1
    neg_token = [0] * (len(word_dict) + 1)
    for word in neg:
        neg_token[word_dict[word]] = 1
    f.write(','.join([str(x) for x in query_token]) + "\t" + ','.join([
        str(x) for x in pos_token
    ]) + "\t" + ','.join([str(x) for x in neg_token]) + "\n")
f.close()

#测试集中的query和pos转化为词袋
f = open("test.txt", "w", encoding="utf-8")
fa = open("label.txt", "w", encoding="utf-8")
for line in test_set:
    query = jieba.cut(line[0].strip())
    pos = jieba.cut(line[1].strip())
    label = line[2]
    query_token = [0] * (len(word_dict) + 1)
    for word in query:
        query_token[word_dict[word]] = 1
    pos_token = [0] * (len(word_dict) + 1)
    for word in pos:
        pos_token[word_dict[word]] = 1
    f.write(','.join([str(x) for x in query_token]) + "\t" + ','.join(
        [str(x) for x in pos_token]) + "\n")
    fa.write(str(label) + "\n")
f.close()
fa.close()
