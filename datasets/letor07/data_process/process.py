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
import numpy as np
import random


# Read Word Dict and Inverse Word Dict
def read_word_dict(filename):
    word_dict = {}
    for line in open(filename):
        line = line.strip().split()
        word_dict[int(line[1])] = line[0]
    print('[%s]\n\tWord dict size: %d' % (filename, len(word_dict)))
    return word_dict


# Read Embedding File
def read_embedding(filename):
    embed = {}
    for line in open(filename):
        line = line.strip().split()
        embed[int(line[0])] = list(map(float, line[1:]))
    print('[%s]\n\tEmbedding size: %d' % (filename, len(embed)))
    return embed


# Convert Embedding Dict 2 numpy array
def convert_embed_2_numpy(embed_dict, embed=None):
    for k in embed_dict:
        embed[k] = np.array(embed_dict[k])
    print('Generate numpy embed:', embed.shape)
    return embed


# Read Data
def read_data(filename):
    data = {}
    for line in open(filename):
        line = line.strip().split()
        data[line[0]] = list(map(int, line[2:]))
    print('[%s]\n\tData size: %s' % (filename, len(data)))
    return data


# Read Relation Data
def read_relation(filename):
    data = []
    for line in open(filename):
        line = line.strip().split()
        data.append((int(line[0]), line[1], line[2]))
    print('[%s]\n\tInstance size: %s' % (filename, len(data)))
    return data


Letor07Path = "./data"
word_dict = read_word_dict(filename=os.path.join(Letor07Path, 'word_dict.txt'))
query_data = read_data(filename=os.path.join(Letor07Path, 'qid_query.txt'))
doc_data = read_data(filename=os.path.join(Letor07Path, 'docid_doc.txt'))
embed_dict = read_embedding(filename=os.path.join(Letor07Path,
                                                  'embed_wiki-pdc_d50_norm'))

_PAD_ = len(word_dict)  #193367
embed_dict[_PAD_] = np.zeros((50, ), dtype=np.float32)
word_dict[_PAD_] = '[PAD]'
W_init_embed = np.float32(np.random.uniform(-0.02, 0.02, [len(word_dict), 50]))
embedding = convert_embed_2_numpy(embed_dict, embed=W_init_embed)
np.save("embedding.npy", embedding)

batch_size = 64
data1_maxlen = 20
data2_maxlen = 500
embed_size = 50
train_iters = 2500


def make_train():
    rel_set = {}
    pair_list = []
    rel = read_relation(filename=os.path.join(Letor07Path,
                                              'relation.train.fold1.txt'))
    for label, d1, d2 in rel:
        if d1 not in rel_set:
            rel_set[d1] = {}
        if label not in rel_set[d1]:
            rel_set[d1][label] = []
        rel_set[d1][label].append(d2)
    for d1 in rel_set:
        label_list = sorted(rel_set[d1].keys(), reverse=True)
        for hidx, high_label in enumerate(label_list[:-1]):
            for low_label in label_list[hidx + 1:]:
                for high_d2 in rel_set[d1][high_label]:
                    for low_d2 in rel_set[d1][low_label]:
                        pair_list.append((d1, high_d2, low_d2))
    print('Pair Instance Count:', len(pair_list))

    f = open("./big_train/train.txt", "w")
    for batch in range(800):
        X1 = np.zeros((batch_size * 2, data1_maxlen), dtype=np.int32)
        X2 = np.zeros((batch_size * 2, data2_maxlen), dtype=np.int32)
        X1[:] = _PAD_
        X2[:] = _PAD_
        for i in range(batch_size):
            d1, d2p, d2n = random.choice(pair_list)
            d1_len = min(data1_maxlen, len(query_data[d1]))
            d2p_len = min(data2_maxlen, len(doc_data[d2p]))
            d2n_len = min(data2_maxlen, len(doc_data[d2n]))
            X1[i, :d1_len] = query_data[d1][:d1_len]
            X2[i, :d2p_len] = doc_data[d2p][:d2p_len]
            X1[i + batch_size, :d1_len] = query_data[d1][:d1_len]
            X2[i + batch_size, :d2n_len] = doc_data[d2n][:d2n_len]
        for i in range(batch_size * 2):
            q = [str(x) for x in list(X1[i])]
            d = [str(x) for x in list(X2[i])]
            f.write(",".join(q) + "\t" + ",".join(d) + "\n")
    f.close()


def make_test():
    rel = read_relation(filename=os.path.join(Letor07Path,
                                              'relation.test.fold1.txt'))
    f = open("./big_test/test.txt", "w")
    for label, d1, d2 in rel:
        X1 = np.zeros(data1_maxlen, dtype=np.int32)
        X2 = np.zeros(data2_maxlen, dtype=np.int32)
        X1[:] = _PAD_
        X2[:] = _PAD_
        d1_len = min(data1_maxlen, len(query_data[d1]))
        d2_len = min(data2_maxlen, len(doc_data[d2]))
        X1[:d1_len] = query_data[d1][:d1_len]
        X2[:d2_len] = doc_data[d2][:d2_len]
        q = [str(x) for x in list(X1)]
        d = [str(x) for x in list(X2)]
        f.write(",".join(q) + "\t" + ",".join(d) + "\t" + str(label) + "\t" +
                d1 + "\n")
    f.close()


make_train()
make_test()
