#coding=utf8
import sys
import random
import json
import numpy as np
import operator
from functools import reduce
from py27hash.hash import hash27

user_fea = ["userid", "gender", "age", "occupation"]
movie_fea = ["movieid", "title", "genres"]
rating_fea = ["userid", "movieid", "rating", "time"]
dict_size = 60000000
hash_dict = dict()

data_path = "data/ml-1m"
test_user_path = "data/online_user"
topk = 100


def read_raw_data():
    user_dict = parse_data(data_path + "/users.dat", user_fea)
    movie_dict = parse_data(data_path + "/movies.dat", movie_fea)
    ratings_dict = dict()
    for line in open(data_path + "/ratings.dat"):
        arr = line.strip().split("::")
        if arr[0] not in ratings_dict:
            ratings_dict[arr[0]] = []
        tmp = dict()
        tmp["movieid"] = arr[1]
        tmp["score"] = arr[2]
        tmp["time"] = arr[3]
        ratings_dict[arr[0]].append(tmp)
    return user_dict, movie_dict, ratings_dict


def parse_data(file_name, feas):
    res = {}
    for line in open(file_name, encoding='ISO-8859-1'):
        line = line.strip()
        arr = line.split("::")
        res[arr[0]] = dict()
        _ = to_hash(feas[0], arr[0])
        for i in range(0, len(feas)):
            res[arr[0]][feas[i]] = arr[i]
    return res


def to_hash(feas, arr):
    out_str = "%s:%s" % (feas, (arr + arr[::-1] + arr[::-2] + arr[::-3]))
    hash_id = hash27(out_str) % dict_size
    if hash_id in hash_dict and hash_dict[hash_id] != out_str:
        print(hash_id, out_str, hash(out_str), hash_dict[hash_id])
        print("conflict")
        exit(-1)
    hash_dict[hash_id] = out_str
    return hash_id


def load_ground_truth(user_dict, movie_dict, ratings_dict):
    for line in open(test_user_path + "/users.dat"):
        uid = line.strip().split("::")[0]
        display_user(user_dict[uid])
        ratings_dict[uid] = sorted(
            ratings_dict[uid],
            key=lambda i: (i["score"], i["time"]),
            reverse=True)
        ratings_dict[uid] = ratings_dict[uid][:topk]
        for i in range(len(ratings_dict[uid])):
            item = ratings_dict[uid][i]
            mid = item["movieid"]
            for key in movie_fea:
                item[key] = movie_dict[mid][key]
        display_movies(ratings_dict[uid])


def load_infer_results(path, feas, movie_dict):
    with open(path) as f:
        content = json.load(f)

    total = 0
    correct = 0
    mae = 0.0

    res = dict()
    for item in content:
        userid = reduce(operator.add, item[feas["userid"]])
        movieid = reduce(operator.add, item[feas["movieid"]])
        ratings = reduce(operator.add, item[feas["ratings"]])
        predict = list(map(int, ratings))
        label = reduce(operator.add, item[feas["label"]])

        mae += sum(np.square(np.array(ratings) - np.array(label)))
        total += len(label)
        correct += sum(np.array(predict) == np.array(label))

        for i in range(len(userid)):
            hash_uid = userid[i]
            hash_mid = movieid[i]
            if hash_uid not in hash_dict or hash_mid not in hash_dict:
                continue
            tmp = hash_dict[hash_uid].split(':')[1]
            uid = tmp[:int(len(tmp) / 3)]
            tmp = hash_dict[hash_mid].split(':')[1]
            mid = tmp[:int(len(tmp) / 3)]
            if uid not in res:
                res[uid] = []
            item = {"score": ratings[i]}
            for info in movie_dict[mid]:
                item[info] = movie_dict[mid][info]
            res[uid].append(item)

    for key in res:
        tmp = sorted(res[key], key=lambda i: i["score"], reverse=True)
        existed_movie = []
        res[key] = []
        for i in range(len(tmp)):
            if len(res[key]) >= topk:
                break
            if tmp[i]["movieid"] not in existed_movie:
                existed_movie.append(tmp[i]["movieid"])
                res[key].append(tmp[i])

    print("total: " + str(total) + "; correct: " + str(correct))
    print("accuracy: " + str(float(correct) / total))
    print("mae: " + str(mae / total))
    return res


def display_user(item):
    out_str = ""
    for key in user_fea:
        out_str += "%s:%s " % (key, item[key])
    print(out_str)


def display_movies(input):
    for item in input:
        print_str = ""
        for key in movie_fea:
            print_str += "%s:%s " % (key, item[key])
        print_str += "%s:%s" % ("score", item["score"])
        print(print_str)


def parse_infer(mode, path, user_dict, movie_dict):
    stage, online = mode.split('_')
    feas = {
        "userid": "userid",
        "movieid": "movieid",
        "ratings": "predict",
        "label": "label"
    }

    infer_results = load_infer_results(path, feas, movie_dict)
    if online.startswith("offline"):
        return

    for uid in infer_results:
        display_user(user_dict[uid])
        display_movies(infer_results[uid])

    with open(test_user_path + "/movies.dat", 'w') as fout:
        for uid in infer_results:
            for item in infer_results[uid]:
                str_ = uid + "::" + str(item["movieid"]) + "::" + str(
                    int(item["score"])) + "\n"
                fout.write(str_)


if __name__ == "__main__":
    user_dict, movie_dict, ratings_dict = read_raw_data()
    if sys.argv[1] == "ground_truth":
        load_ground_truth(user_dict, movie_dict, ratings_dict)
    else:
        parse_infer(sys.argv[1], sys.argv[2], user_dict, movie_dict)
