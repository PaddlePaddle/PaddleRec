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

import time
import numpy as np
import sys
import os

import argparse
import json
import random
import multiprocessing as mp


def mp_run(data, process_num, func, *args):
    """ run func with multi process
    """
    level_start = time.time()
    partn = int(max(len(data) / process_num, 1))
    start = 0
    p_idx = 0
    ps = []
    while start < len(data):
        local_data = data[start:start + partn]
        start += partn
        p = mp.Process(target=func, args=(local_data, p_idx) + args)
        ps.append(p)
        p.start()
        p_idx += 1
    for p in ps:
        p.join()

    for p in ps:
        p.terminate()
    return p_idx


def read(train_data_file, test_data_file):
    behavior_dict = dict()
    train_sample = dict()
    test_sample = dict()
    user_id = list()
    item_id = list()
    cat_id = list()
    behav_id = list()
    timestamp = list()

    start = time.time()
    itobj = zip([train_data_file, test_data_file], [train_sample, test_sample])
    for filename, sample in itobj:
        with open(filename, 'r') as f:
            for line in f:
                arr = line.strip().split(',')
                if len(arr) != 5:
                    break
                user_id.append(int(arr[0]))
                item_id.append(int(arr[1]))
                cat_id.append(int(arr[2]))
                if arr[3] not in behavior_dict:
                    i = len(behavior_dict)
                    behavior_dict[arr[3]] = i
                behav_id.append(behavior_dict[arr[3]])
                timestamp.append(int(arr[4]))
            sample["USERID"] = np.array(user_id)
            sample["ITEMID"] = np.array(item_id)
            sample["CATID"] = np.array(cat_id)
            sample["BEHAV"] = np.array(behav_id)
            sample["TS"] = np.array(timestamp)

            user_id = []
            item_id = []
            cat_id = []
            behav_id = []
            timestamp = []

    print("Read data done, {} train records, {} test records"
          ", elapsed: {}".format(
              len(train_sample["USERID"]),
              len(test_sample["USERID"]), time.time() - start))
    return behavior_dict, train_sample, test_sample


def gen_user_his_behave(train_sample):
    user_his_behav = dict()
    iterobj = zip(train_sample["USERID"], train_sample["ITEMID"],
                  train_sample["TS"])
    for user_id, item_id, ts in iterobj:
        if user_id not in user_his_behav:
            user_his_behav[user_id] = list()
        user_his_behav[user_id].append((item_id, ts))

    for _, value in user_his_behav.items():
        value.sort(key=lambda x: x[1])

    return user_his_behav


def split_train_sample(train_dir, train_sample_seg_cnt):
    segment_filenames = []
    segment_files = []
    for i in range(train_sample_seg_cnt):
        filename = "{}/part_{}".format(train_dir, i)
        segment_filenames.append(filename)
        segment_files.append(open(filename, 'w'))

    with open("train_tmp", 'r') as fi:
        for line in fi:
            i = random.randint(0, train_sample_seg_cnt - 1)
            segment_files[i].write(line)

    for f in segment_files:
        f.close()

    os.remove("train_tmp")

    # Shuffle
    for fn in segment_filenames:
        lines = []
        with open(fn, 'r') as f:
            for line in f:
                lines.append(line)
        random.shuffle(lines)
        with open(fn, 'w') as f:
            for line in lines:
                f.write(line)


def partial_gen_train_sample(users, user_his_behav, filename, pipe, seq_len,
                             min_len):
    stat = dict()
    with open(filename, 'w') as f:
        for user in users:
            value = user_his_behav[user]
            count = len(value)
            if count < min_len:
                continue
            arr = [0 for i in range(seq_len - min_len)] + \
                    [v[0] for v in value]
            for i in range(len(arr) - seq_len + 1):
                sample = arr[i:i + seq_len]
                f.write('{}_{}'.format(user, i))  # sample id
                f.write('\t{}'.format(sample[-1]))  # label feature
                for j in range(seq_len - 1):
                    if sample[j] != 0:
                        f.write("\tslot_{}:{}".format(j + 1, sample[j]))
                f.write('\n')
                if sample[-1] not in stat:
                    stat[sample[-1]] = 0
                stat[sample[-1]] += 1
    pipe.send(stat)


def gen_train_sample(train_sample, args):
    user_his_behav = gen_user_his_behave(train_sample)
    print("user_his_behav len: {}".format(len(user_his_behav)))

    users = list(user_his_behav.keys())
    process = []
    pipes = []
    parall = args.parall
    job_size = int(len(user_his_behav) / parall)
    if len(user_his_behav) % parall != 0:
        parall += 1
    for i in range(parall):
        a, b = mp.Pipe()
        pipes.append(a)
        p = mp.Process(
            target=partial_gen_train_sample,
            args=(users[i * job_size:(i + 1) * job_size], user_his_behav,
                  'train_tmp.part_{}'.format(i), b, args.seq_len,
                  args.min_seq_len))
        process.append(p)
        p.start()

    stat = dict()
    for pipe in pipes:
        st = pipe.recv()
        for k, v in st.items():
            k = int(k)
            if k not in stat:
                stat[k] = 0
            stat[k] += v

    for p in process:
        p.join()

    # Merge partial files
    with open("train_tmp", 'w') as f:
        for i in range(parall):
            filename = 'train_tmp.part_{}'.format(i)
            with open(filename, 'r') as f1:
                f.write(f1.read())

            os.remove(filename)

    # Split train sample to segments
    split_train_sample(args.train_dir, args.train_sample_seg_cnt)
    return stat


def gen_test_sample(test_dir, test_sample, seq_len, min_seq_len):
    user_his_behav = gen_user_his_behave(test_sample)
    with open("{}/part-0".format(test_dir), 'w') as f:
        for user, value in user_his_behav.items():
            if len(value) / 2 + 1 < min_seq_len:
                continue

            mid = int(len(value) / 2)
            left = value[:mid][-seq_len + 1:]
            right = value[mid:]
            left = [0 for i in range(seq_len - len(left) - 1)] + \
                    [l[0] for l in left]
            f.write('{}_{}'.format(user, 'T'))  # sample id
            labels = ','.join(map(str, [item[0] for item in right]))
            f.write('\t{}'.format(labels))
            # kvs
            for j in range(seq_len - 1):
                if left[j] != 0:
                    f.write("\tslot_{}:{}".format(j + 1, left[j]))
            f.write('\n')


def prepare_sample_set(train_dir, sample_dir, process_num=12, feature_num=69):
    def parse_data(files, idx, feature_num=69):
        history_ids = [0] * feature_num
        samples = dict()
        process = 0
        for filename in files:
            process += 1
            print("process {} / {}.".format(process, len(files)))
            with open(filename) as f:
                print("Begin to handle {}.".format(filename))
                for line in f:
                    history_ids = [0] * feature_num
                    features = line.strip().split("\t")
                    item_id = int(features[1])
                    for item in features[2:]:
                        slot, feasign = item.split(":")
                        slot_id = int(slot.split("_")[1])
                        history_ids[slot_id - 1] = int(feasign)
                    if item_id not in samples:
                        samples[item_id] = list()
                    samples[item_id].append(history_ids)

        with open("parse_data_{}.json".format(idx), 'w') as json_file:
            json.dump(samples, json_file)

    files = ["{}/{}".format(train_dir, f) for f in os.listdir(train_dir)]
    real_process_num = mp_run(files, process_num, parse_data, feature_num)

    num = 0
    all_samples = dict()
    for i in range(real_process_num):
        filename = "parse_data_{}.json".format(i)
        with open(filename, 'r') as json_file:
            each_samples = json.load(json_file)
            for key in each_samples:
                if key not in all_samples:
                    all_samples[key] = []
                all_samples[key] += each_samples[key]
                num += len(each_samples[key])
        os.remove(filename)

    for ck in all_samples:
        with open("{}/samples_{}.json".format(sample_dir, ck), 'w') as f:
            json.dump(all_samples[ck], f)


if __name__ == '__main__':
    _PARSER = argparse.ArgumentParser(description="DataProcess")
    _PARSER.add_argument("--train_file", required=True, help="Train filename")
    _PARSER.add_argument("--test_file", required=True, help="Test filename")
    _PARSER.add_argument(
        "--item_cate_filename",
        default="./Item_Cate.txt",
        help="item cate filename, used to init the first tree.")
    _PARSER.add_argument(
        "--stat_file", default="./Stat.txt", help="Stat filename")

    _PARSER.add_argument(
        "--train_dir", default="./train_data", help="Train directory")
    _PARSER.add_argument(
        "--sample_dir", default="./samples", help="Sample directory")
    _PARSER.add_argument(
        "--test_dir", default="./test_data", help="Test directory")

    _PARSER.add_argument(
        '--parall', type=int, help="parall process used", default=16)
    _PARSER.add_argument(
        "--train_sample_seg_cnt",
        type=int,
        default=20,
        help="count of train sample segments file")
    _PARSER.add_argument(
        "--seq_len",
        type=int,
        help="sequence length of the sample record",
        default=70)
    _PARSER.add_argument(
        "--min_seq_len",
        type=int,
        help="Min length of the sample sequence record",
        default=8)
    args = _PARSER.parse_args()

    os.system("rm -rf ./{} && mkdir -p {}".format(args.train_dir,
                                                  args.train_dir))
    os.system("rm -rf ./{} && mkdir -p {}".format(args.test_dir,
                                                  args.test_dir))
    os.system("rm -rf ./{} && mkdir -p {}".format(args.sample_dir,
                                                  args.sample_dir))

    behavior_dict, train_sample, test_sample = read(args.train_file,
                                                    args.test_file)
    print(repr(behavior_dict))
    stat = gen_train_sample(train_sample, args)
    with open(args.stat_file, 'w') as f:
        json.dump(stat, f)
    gen_test_sample(args.test_dir, test_sample, args.seq_len, args.min_seq_len)

    item_cate = dict()
    for sample in [train_sample, test_sample]:
        iterobj = zip(sample["ITEMID"], sample["CATID"])
        for item_id, cat_id in iterobj:
            if item_id not in item_cate:
                item_cate[item_id] = cat_id

    with open(args.item_cate_filename, 'w') as f:
        for key in item_cate:
            f.write("{}\t{}\n".format(key, item_cate[key]))

    prepare_sample_set(
        args.train_dir,
        args.sample_dir,
        args.parall,
        feature_num=args.seq_len - 1)
