#encoding=utf-8

import os
import sys
import numpy as np
import random

f = open("./zhidao", "r")
lines = f.readlines()
f.close()

#建立字典
word_dict = {}
for line in lines:
    line = line.strip().split("\t")
    text = line[0].split(" ") + line[1].split(" ")
    for word in text:
        if word in word_dict:
            word_dict[word] = word_dict[word] + 1
        else:
            word_dict[word] = 1

word_list = word_dict.items()
word_list = sorted(word_dict.items(), key=lambda item: item[1], reverse=True)
word_list_ids = range(1, len(word_list) + 1)
word_dict = dict(zip([x[0] for x in word_list], word_list_ids))

f = open("./zhidao", "r")
lines = f.readlines()
f.close()

#划分训练集和测试集
lines = [line.strip().split("\t") for line in lines]
random.shuffle(lines)
train_set = lines[:900]
test_set = lines[900:]

#建立以query为key，以负例为value的字典
neg_dict = {}
for line in train_set:
    if line[2] == "0":
        if line[0] in neg_dict:
            neg_dict[line[0]].append(line[1])
        else:
            neg_dict[line[0]] = [line[1]]

#建立以query为key，以正例为value的字典
pos_dict = {}
for line in train_set:
    if line[2] == "1":
        if line[0] in pos_dict:
            pos_dict[line[0]].append(line[1])
        else:
            pos_dict[line[0]] = [line[1]]

#训练集整理为query，pos，neg的格式
f = open("train.txt", "w")
for query in pos_dict.keys():
    for pos in pos_dict[query]:
        if query not in neg_dict:
            continue
        for neg in neg_dict[query]:
            f.write(str(query) + "\t" + str(pos) + "\t" + str(neg) + "\n")
f.close()

f = open("train.txt", "r")
lines = f.readlines()
f.close()

#训练集中的query,pos,neg转化格式
f = open("train.txt", "w")
for line in lines:
    line = line.strip().split("\t")
    query = line[0].strip().split(" ")
    pos = line[1].strip().split(" ")
    neg = line[2].strip().split(" ")
    query_list = []
    for word in query:
        query_list.append(word_dict[word])
    pos_list = []
    for word in pos:
        pos_list.append(word_dict[word])
    neg_list = []
    for word in neg:
        neg_list.append(word_dict[word])
    f.write(' '.join(["0:" + str(x) for x in query_list]) + " " + ' '.join([
        "1:" + str(x) for x in pos_list
    ]) + " " + ' '.join(["2:" + str(x) for x in neg_list]) + "\n")
f.close()

#测试集中的query和pos转化格式
f = open("test.txt", "w")
fa = open("label.txt", "w")
for line in test_set:
    query = line[0].strip().split(" ")
    pos = line[1].strip().split(" ")
    label = line[2]
    query_list = []
    for word in query:
        query_list.append(word_dict[word])
    pos_token = []
    for word in pos:
        pos_list.append(word_dict[word])
    f.write(' '.join(["0:" + str(x) for x in query_list]) + " " + ' '.join(
        ["1:" + str(x) for x in pos_list]) + "\n")
    fa.write(label + "\n")

f.close()
fa.close()
