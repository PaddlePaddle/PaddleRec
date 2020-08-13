# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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


class Dataset:
    def __init__(self):
        pass


class SyntheticDataset(Dataset):
    def __init__(self,
                 sparse_feature_dim,
                 query_slot_num,
                 title_slot_num,
                 dataset_size=10000):
        # ids are randomly generated
        self.ids_per_slot = 10
        self.sparse_feature_dim = sparse_feature_dim
        self.query_slot_num = query_slot_num
        self.title_slot_num = title_slot_num
        self.dataset_size = dataset_size

    def _reader_creator(self, is_train):
        def generate_ids(num, space):
            return [random.randint(0, space - 1) for i in range(num)]

        def reader():
            for i in range(self.dataset_size):
                query_slots = []
                pos_title_slots = []
                neg_title_slots = []
                for i in range(self.query_slot_num):
                    qslot = generate_ids(self.ids_per_slot,
                                         self.sparse_feature_dim)
                    qslot = [str(fea) + ':' + str(i) for fea in qslot]
                    query_slots += qslot
                for i in range(self.title_slot_num):
                    pt_slot = generate_ids(self.ids_per_slot,
                                           self.sparse_feature_dim)
                    pt_slot = [
                        str(fea) + ':' + str(i + self.query_slot_num)
                        for fea in pt_slot
                    ]
                    pos_title_slots += pt_slot
                if is_train:
                    for i in range(self.title_slot_num):
                        nt_slot = generate_ids(self.ids_per_slot,
                                               self.sparse_feature_dim)
                        nt_slot = [
                            str(fea) + ':' +
                            str(i + self.query_slot_num + self.title_slot_num)
                            for fea in nt_slot
                        ]
                        neg_title_slots += nt_slot
                    yield query_slots + pos_title_slots + neg_title_slots
                else:
                    yield query_slots + pos_title_slots

        return reader

    def train(self):
        return self._reader_creator(True)

    def valid(self):
        return self._reader_creator(True)

    def test(self):
        return self._reader_creator(False)


if __name__ == '__main__':
    sparse_feature_dim = 1000001
    query_slots = 1
    title_slots = 1
    dataset_size = 10
    dataset = SyntheticDataset(sparse_feature_dim, query_slots, title_slots,
                               dataset_size)
    train_reader = dataset.train()
    test_reader = dataset.test()

    with open("data/train/train.txt", 'w') as fout:
        for data in train_reader():
            fout.write(' '.join(data))
            fout.write("\n")

    with open("data/test/test.txt", 'w') as fout:
        for data in test_reader():
            fout.write(' '.join(data))
            fout.write("\n")
