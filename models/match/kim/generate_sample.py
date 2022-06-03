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

import pandas as pd
import json
import numpy as n

base = '/home/aistudio/paddlrec/datasets/kim/data/whole_data/'
base2 = '/home/aistudio/paddlrec/datasets/kim/data/sample_data/'
os.makedirs(base2, exist_ok=True)
KGGraph = base + 'KGGraph.json'

doc = pd.read_csv(base + 'docs.tsv', sep='\t', header=None)

doc_names = doc[0].tolist()
train_file = base2 + 'train.tsv'
train = pd.read_csv(train_file, sep='\t', header=None)
train = train.dropna()
train.to_csv(train_file, sep='\t', header=None, index=None)
test_file = base2 + 'test.tsv'
test = pd.read_csv(test_file, sep='\t', header=None)
test = test.dropna()
test.to_csv(test_file, sep='\t', header=None, index=None)
train = pd.concat([train, test])
used = set()
for i in train[3]:
    used |= set(i.split())
for i in train[4]:
    for j in i.split():
        used.add(j.split('-')[0])
for i in used:
    assert i in doc_names

doc = doc[doc[0].isin(used)]
# doc.to_csv(
#     '/home/xianglingyang/project/paddle_family/paddle_reproduction/PaddleRec/models/recall/kim/data/sample_data/docs.tsv',
#     sep='\t', header=None, index=None)

with open(base2 + 'docs.tsv', 'w') as f:
    for _, row in doc.iterrows():
        # print(row)
        f.write('\t'.join(row.fillna('')) + '\n')

entities = set()
for i in doc.values[:, -2]:
    i = json.loads(i)
    for j in i:
        entities.add(j['WikidataId'])
with open(KGGraph, 'r') as f:
    KGGraph = json.load(f)

new_kg = {}
new_entities = []
for i in entities:
    if i in KGGraph:
        new_kg[i] = KGGraph[i][:10]
        new_entities.append(i)
        new_entities.extend(KGGraph[i][:20])
# new_kg = {x: KGGraph[x] for x in entities if x in KGGraph}
with open(base2 + 'KGGraph.json', 'w') as f:
    json.dump(new_kg, f)

entities = sorted(set(new_entities))
n = len(new_entities)
with open(base2 + 'entity2id.txt', 'w') as f:
    f.write(str(n) + '\n')
    for i, v in enumerate(new_entities):
        f.write('{}\t{}\n'.format(v, i))

import numpy as np

emb = np.load(base + 'entity_embedding.npy')
np.save(base2 + 'entity_embedding.npy', emb[:n])

# print(line[3])
# print(line[4])
# for i in line[3].split():
#     assert i in doc_names

# print(doc.head())
