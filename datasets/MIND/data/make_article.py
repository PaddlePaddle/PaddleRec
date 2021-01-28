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

import random
cate_dict = {}
sub_cate_dict = {}
with open("cate_map", "r") as r:
    for l in r:
        line = l.split("\t")
        cate_dict[line[0]] = line[1][:-1]

with open("sub_cate_map", "r") as r:
    for l in r:
        line = l.split("\t")
        sub_cate_dict[line[0]] = line[1][:-1]

print(cate_dict)
print(sub_cate_dict)
title_r = []
content_r = []
index = 0
with open("convert_text8/convert_temp.txt", "r") as r1:
    for l in r1:
        if index % 2 == 0:
            title_r.append(l[:-1])
        else:
            content_r.append(l[:-1])
        index = index + 1
print(index)
inx = 0
article_map = {}
with open("article.txt", "w") as ar:
    with open("news_backup.tsv", "r") as r2:
        for l in r2:
            line = l.split("\t")
            id, cate, sub_cate = line[:3]
            article_map[id] = len(article_map)
            cate = cate_dict[cate]
            sub_cate = sub_cate_dict[sub_cate]
            title = title_r[inx]
            content = content_r[inx]
            inx = inx + 1
            ar.write(id + "\t" + cate + "\t" + sub_cate + "\t" + title + "\t" +
                     content + "\n")

print(inx)
print(len(article_map))
file_base = ["train", "dev"]
files = ["train_raw/behaviors.tsv", "dev_raw/behaviors.tsv"]
for base, aim in zip(files, file_base):
    with open(aim + "/browse.txt", "w") as w1:
        print("generate " + aim)
        with open(base, "r") as f:
            for l in f:
                line = l.split("\t")
                visit = line[3]
                if len(visit) == 0:
                    continue
                sample = line[4].split(" ")
                pos_sample = ""
                neg_sample = ""
                for s in sample:
                    if len(s) > 0 and (s[-2:] == "-1" or s[-2:] == "-0"):
                        id = s.split("-")[0]
                        if id in article_map:
                            if s[-2:] == "-1":
                                if len(pos_sample) > 0:
                                    pos_sample = pos_sample + " "
                                pos_sample = pos_sample + id
                            else:
                                if len(neg_sample) > 0:
                                    neg_sample = neg_sample + " "
                                neg_sample = neg_sample + id

                if len(pos_sample) == 0 or len(neg_sample) == 0:
                    continue

                line = visit + "\t" + pos_sample + "\t" + neg_sample + "\n"
                w1.write(line)


def remove(str, sp1, sp2):
    l = list(str)
    index2 = -1
    for i in range(len(l) - 1, -1, -1):
        if l[i] == sp1:
            index2 = i
            break
    if index2 == -1:
        return '', False
    l[index2] = sp2
    return ''.join(l), True


with open("test_raw/behaviors.tsv", "r") as r:
    with open("test/browse.txt", "w") as w:
        for l in r:
            x, y = l.split('\t')[3:5]
            line2, stat = remove(x, ' ', '\t')
            if stat == True:
                w.write(line2 + "\t" + y)
