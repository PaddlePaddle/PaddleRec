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

#划分训练集和测试集
query_list = list(pos_dict.keys())
#print(len(query))
random.shuffle(query_list)
train_query = query_list[:90]
test_query = query_list[90:]

#获得训练集
train_set = []
for query in train_query:
    for pos in pos_dict[query]:
        if query not in neg_dict:
            continue
        for neg in neg_dict[query]:
            train_set.append([query, pos, neg])
random.shuffle(train_set)

#获得测试集
test_set = []
for query in test_query:
    for pos in pos_dict[query]:
        test_set.append([query, pos, 1])
    if query not in neg_dict:
        continue
    for neg in neg_dict[query]:
        test_set.append([query, pos, 0])
random.shuffle(test_set)

#训练集中的query,pos,neg转化为词袋
f = open("train.txt", "w")
f = open("train.txt", "w")
for line in train_set:
    query = line[0].strip().split(" ")
    pos = line[1].strip().split(" ")
    neg = line[2].strip().split(" ")
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
f = open("test.txt", "w")
fa = open("label.txt", "w")
for line in test_set:
    query = line[0].strip().split(" ")
    pos = line[1].strip().split(" ")
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
