# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import pandas as pd
import argparse


def gen_heter_files(data_load_path, splitted_data_path, file_nums):
    data = pd.read_csv(data_load_path)
    total_sample_num = data.shape[0]
    print("total sample num is: {}".format(total_sample_num))
    sample_num_per_file = int(total_sample_num / file_nums)
    for i in range(0, file_nums):
        save_data = data.iloc[i * sample_num_per_file + 1:(i + 1) *
                              sample_num_per_file + 1]
        file_name = splitted_data_path + '/' + str(i) + '.csv'
        save_data.to_csv(file_name, index=False)
    print("files splitted done, num is {}, saved in path: {}".format(
        file_nums, splitted_data_path))


def get_zipcode_dict():
    filename = '/home/wangbin/the_one_ps/ziyoujiyi_PaddleRec/MovieLens-1M/ml-1m/users.dat'
    zipcode_dict = {}
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("::")
            user_id, sex, age, occupation, zip_code = int(arr[0]), str(arr[
                1]), int(arr[2]), int(arr[3]), str(arr[4])
            zip_code = int(zip_code[0:5])
            zipcode_dict[user_id] = zip_code
            line = f.readline()
    return zipcode_dict


def shuffle_data_by_zipcode(data_load_path, splitted_data_path, file_nums,
                            zipcode_dict):
    data = pd.read_csv(data_load_path)
    total_sample_num = data.shape[0]
    print("total sample num is: {}".format(total_sample_num))
    data_list = data.values.tolist()
    sharded_data = [(idx, []) for idx in range(10)]
    for data_row in data_list:
        user_id = data_row[0]
        zipcode = zipcode_dict[user_id + 1]
        shard_id = int(zipcode / 10000)
        sharded_data[shard_id][1].extend([data_row])
    for (shard_id, sample) in sharded_data:
        print("zipcode start with {}: {}".format(shard_id, len(sample)))
        file_name = splitted_data_path + '/' + str(shard_id) + '.csv'
        d = pd.DataFrame(data=sample)
        d.to_csv(file_name, index=False)
    print("files splitted by zipcode done, saved in path: {}".format(
        splitted_data_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument(
        '--full_train_data_path',
        type=str,
        default="../big_train/train_data.csv",
        help='full_train_data_path')
    parser.add_argument(
        '--splitted_data_path',
        type=str,
        default="fl_train_data",
        help='splitted_data_path')
    parser.add_argument(
        '--file_nums', type=int, default='10', help='fl clients num')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    #gen_heter_files(args.full_train_data_path, args.splitted_data_path, args.file_nums)
    zipcode_dict = get_zipcode_dict()
    shuffle_data_by_zipcode(args.full_train_data_path, args.splitted_data_path,
                            args.file_nums, zipcode_dict)
