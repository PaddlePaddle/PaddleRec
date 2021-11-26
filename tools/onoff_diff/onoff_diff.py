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

#!/usr/bin/env python
# coding=utf-8

import os
import sys
import re
import json
import numpy

FLAG = True

#FLAG=False


def get_all_vars(var_file):
    var_list = []

    with open(var_file, 'r') as fin:
        for line in fin:
            line = line.strip('\n')
            if len(line) > 0:
                var_list.append(line)

    return var_list


def get_data_from_log(online_log_file, origin_file):
    insid_list = []
    with open(origin_file, 'r') as fin:
        for line in fin:
            insid = line.strip('\n').split('\t')[0]
            insid_list.append(insid)
    # print("insid_list:", insid_list)

    res = {}
    pattern = r'\[([^\]]*)\]'
    with open(online_log_file, 'r') as fin:
        for line in fin:
            insid = ''
            var_name = ''
            length = 0
            data = []
            ele_list = re.findall(pattern, line.strip('\n'))
            for ele in ele_list:
                if ele.find(':') < 0:
                    continue
                key, value = ele.split(':')
                if key == 'batchIdx':
                    insid = insid_list[int(value)]
                if key == 'name':
                    var_name = value
                if key == 'shape':
                    dim1, dim2 = value.split(',')
                    length = int(dim1) * int(dim2)
                if key == 'data':
                    data = [float(x) for x in value.split(',')]
            assert length == len(data)
            if insid not in res:
                res[insid] = {}
            res[insid][var_name] = (length, data)
    # print("online log:", res.keys())

    return res


def get_data_from_model(model_dump_file):
    res = {}

    with open(model_dump_file, 'r') as fin:
        for line in fin:
            line = line.strip('\n').split('\t')
            insid = line[0]
            res[insid] = {}
            for var in line[2:]:
                var_name, length, data = var.split(':', 2)
                length = int(length)
                data = [float(x) for x in data.split(':')]
                assert length == len(data)
                res[insid][var_name] = (length, data)
    # print("offline log:", res.keys())
    return res


def onoff_var_diff(log_data, model_data, var_name):
    res = {'labels': [], 'values': []}
    diff_ins_list = []

    same_insid_counts = 0
    all_diff = []
    for insid in log_data.keys():
        if insid in model_data:
            same_insid_counts += 1
            log_val = log_data[insid][var_name]
            if var_name in model_data[insid]:
                model_val = model_data[insid][var_name]
                assert log_val[0] == model_val[0]
                for i in range(log_val[0]):
                    all_diff.append(abs(log_val[1][i] - model_val[1][i]))
                    print(insid, log_val[1][i], model_val[1][i])
                    if abs(log_val[1][i] - model_val[1][i]) > 0.01:
                        diff_ins_list.append((insid, log_val[1], model_val[1]))
    print("diff_ins_list:", diff_ins_list)
    # print("all_diff:", all_diff)
    all_diff = numpy.asarray(all_diff)
    total = same_insid_counts
    print("same ins total: {}".format(same_insid_counts))
    no_diff = numpy.sum(all_diff < 0.000001)
    if no_diff:
        res['labels'].append('无diff')
        res['values'].append(no_diff)
        print("无diff: {}, {}".format(no_diff, float(no_diff) / total))
    labels = [
        '个位diff', '十分位diff', '百分位diff', '千分位diff', '万分位diff', '十万分位diff',
        '百万分位diff'
    ]
    multis = [1, 10, 100, 1000, 10000, 100000, 1000000]
    for label, multi in zip(labels, multis):
        diff_counts = numpy.sum(all_diff * multi > 1)
        if diff_counts:
            res['labels'].append(label)
            res['values'].append(diff_counts)
            print("{}: {}, {}".format(label, diff_counts,
                                      float(diff_counts) / total))
    # return json.dumps(res, ensure_ascii=False) if res['values'] else ""
    return diff_ins_list


def onoff_max_diff(log_data, model_data, ins_id):
    log_ins = log_data[ins_id]
    model_ins = model_data[ins_id]
    for var_name in model_ins:
        max_diff = 0.0
        if var_name not in log_ins:
            print(
                'var {} is not in online log'.format(var_name),
                file=sys.stderr)
            continue
        if log_ins[var_name][0] != model_ins[var_name][0]:
            print(
                'The length of {} is wrong, online is {}, offline is {}'.
                format(var_name, log_ins[var_name][0], model_ins[var_name][0]),
                file=sys.stderr)
            continue
        for i in range(log_ins[var_name][0]):
            diff = abs(log_ins[var_name][1][i] - model_ins[var_name][1][i])
            if max_diff < diff:
                max_diff = diff
        if max_diff > 2e-5:
            print("ins_id:{}, var_name:{}: {}".format(ins_id, var_name,
                                                      max_diff))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', help='线上日志文件', dest='online_log_file', required=True)
    parser.add_argument(
        '-m', help='模型dump文件', dest='model_dump_file', required=True)
    parser.add_argument('-v', help='所有var文件', dest='var_file', required=True)
    parser.add_argument(
        '-o', help='offline原始数据文件', dest='origin_file', required=True)
    args = parser.parse_args()
    log_data = get_data_from_log(args.online_log_file, args.origin_file)
    model_data = get_data_from_model(args.model_dump_file)
    var_list = get_all_vars(args.var_file)
    diff_ins_list = onoff_var_diff(log_data, model_data, var_list[-1])
    # diff_ins_list = [str(i) for i in range(1, 11)]
    for ins in diff_ins_list:
        onoff_max_diff(log_data, model_data, ins)
    # print('<pie-charts>{}</pie-charts>'.format(res))
