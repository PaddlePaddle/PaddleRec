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

#coding=utf8
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import random
import json

user_fea = ["userid", "gender", "age", "occupation"]
movie_fea = ["movieid", "title", "genres"]
rating_fea = ["userid", "movieid", "rating", "time"]
dict_size = 1000000
hash_dict = dict()

data_path = "ml-1m"
test_user_path = "online_user"


def process(path, output_path):
    user_dict = parse_data(data_path + "/users.dat", user_fea)
    movie_dict = parse_movie_data(data_path + "/movies.dat", movie_fea)

    res = []
    for line in open(path):
        line = line.strip()
        arr = line.split("::")
        userid = arr[0]
        movieid = arr[1]
        out_str = "time:%s\t%s\t%s\tlabel:%s" % (arr[3], user_dict[userid],
                                                 movie_dict[movieid], arr[2])
        log_id = hash(out_str) % 1000000000
        res.append("%s\t%s" % (log_id, out_str))
    with open(output_path, 'w') as fout:
        for line in res:
            fout.write(line)
            fout.write("\n")


def parse_data(file_name, feas):
    dict = {}
    for line in open(file_name):
        line = line.strip()
        arr = line.split("::")
        out_str = ""
        for i in range(0, len(feas)):
            out_str += "%s:%s\t" % (feas[i], arr[i])

        dict[arr[0]] = out_str.strip()
    return dict


def parse_movie_data(file_name, feas):
    dict = {}
    for line in open(file_name):
        line = line.strip()
        arr = line.split("::")
        title_str = ""
        genres_str = ""

        for term in arr[1].split(" "):
            term = term.strip()
            if term != "":
                title_str += "%s " % (term)
        for term in arr[2].split("|"):
            term = term.strip()
            if term != "":
                genres_str += "%s " % (term)
        out_str = "movieid:%s\ttitle:%s\tgenres:%s" % (
            arr[0], title_str.strip(), genres_str.strip())
        dict[arr[0]] = out_str.strip()
    return dict


def to_hash(in_str):
    feas = in_str.split(":")[0]
    arr = in_str.split(":")[1]
    out_str = "%s:%s" % (feas, (arr + arr[::-1] + arr[::-2] + arr[::-3]))
    hash_id = hash(out_str) % dict_size
    #  if hash_id in hash_dict and hash_dict[hash_id] != out_str:
    #      print(hash_id, out_str, hash(out_str))
    #      print("conflict")
    #  exit(-1)

    return "%s:%s" % (feas, hash_id)


def to_hash_list(in_str):
    arr = in_str.split(":")
    tmp_arr = arr[1].split(" ")
    out_str = ""
    for item in tmp_arr:
        item = item.strip()
        if item != "":
            key = "%s:%s" % (arr[0], item)
            out_str += "%s " % (to_hash(key))
    return out_str.strip()


def get_hash(path):
    #0-34831 1-time:974673057 2-userid:2021 3-gender:M 4-age:25 5-occupation:0 6-movieid:1345  7-title:Carrie (1976)  8-genres:Horror  9-label:2
    for line in open(path):
        arr = line.strip().split("\t")
        out_str = "logid:%s %s %s %s %s %s %s %s %s %s" % \
                 (arr[0], arr[1], to_hash(arr[2]), to_hash(arr[3]), to_hash(arr[4]), to_hash(arr[5]), \
                 to_hash(arr[6]), to_hash_list(arr[7]), to_hash_list(arr[8]), arr[9])
        print out_str


def split(path, output_dir, num=24):
    contents = []
    with open(path) as f:
        contents = f.readlines()
    lines_per_file = len(contents) / num
    print("contents: ", str(len(contents)))
    print("lines_per_file: ", str(lines_per_file))

    for i in range(1, num + 1):
        with open(os.path.join(output_dir, "part_" + str(i)), 'w') as fout:
            data = contents[(i - 1) * lines_per_file:min(i * lines_per_file,
                                                         len(contents))]
            for line in data:
                fout.write(line)


if __name__ == "__main__":
    random.seed(1111111)
    if sys.argv[1] == "process_raw":
        process(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "hash":
        get_hash(sys.argv[2])
    elif sys.argv[1] == "split":
        split(sys.argv[2], sys.argv[3])
