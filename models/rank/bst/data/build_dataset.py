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

from __future__ import print_function
import random
import pickle

random.seed(1234)

print("read and process data")

with open('./raw_data/remap.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
    pos_list = hist['asin'].tolist()
    time_list = hist['unixReviewTime'].tolist()

    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count - 1)
        return neg

    neg_list = [gen_neg() for i in range(len(pos_list))]

    for i in range(1, len(pos_list)):
        hist = pos_list[:i]
        # set maximum position value
        time_seq = [
            min(int((time_list[i] - time_list[j]) / (3600 * 24)), 5000)
            for j in range(i)
        ]
        if i != len(pos_list) - 1:
            train_set.append((reviewerID, hist, pos_list[i], 1, time_seq))
            train_set.append((reviewerID, hist, neg_list[i], 0, time_seq))
        else:
            label = (pos_list[i], neg_list[i])
            test_set.append((reviewerID, hist, label, time_seq))

random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count


def print_to_file(data, fout, slot):
    if not isinstance(data, list):
        data = [data]
    for i in range(len(data)):
        fout.write(slot + ":" + str(data[i]))
        fout.write(' ')


print("make train data")
with open("./all_data/train_data/paddle_train.txt", "w") as fout:
    for line in train_set:
        userid = line[0]
        history = line[1]
        target = line[2]
        label = line[3]
        position = line[4]
        cate = [cate_list[x] for x in history]
        print_to_file(userid, fout, "userid")
        print_to_file(history, fout, "history")
        print_to_file(cate, fout, "cate")
        print_to_file(position, fout, "position")
        print_to_file(target, fout, "target")
        print_to_file(cate_list[target], fout, "target_cate")
        print_to_file(0, fout, "target_position")
        print_to_file(label, fout, "label")
        fout.write("\n")

print("make test data")
with open("./all_data/test_data/paddle_test.txt", "w") as fout:
    for line in test_set:
        userid = line[0]
        history = line[1]
        target = line[2]
        position = line[3]
        cate = [cate_list[x] for x in history]

        print_to_file(userid, fout, "userid")
        print_to_file(history, fout, "history")
        print_to_file(cate, fout, "cate")
        print_to_file(position, fout, "position")
        print_to_file(target[0], fout, "target")
        print_to_file(cate_list[target[0]], fout, "target_cate")
        print_to_file(0, fout, "target_position")
        fout.write("label:1\n")

        print_to_file(userid, fout, "userid")
        print_to_file(history, fout, "history")
        print_to_file(cate, fout, "cate")
        print_to_file(position, fout, "position")
        print_to_file(target[1], fout, "target")
        print_to_file(cate_list[target[1]], fout, "target_cate")
        print_to_file(0, fout, "target_position")
        fout.write("label:0\n")

print("make config data")
with open('config.txt', 'w') as f:
    f.write(str(user_count) + "\n")
    f.write(str(item_count) + "\n")
    f.write(str(cate_count) + "\n")
