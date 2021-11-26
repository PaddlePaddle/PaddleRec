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
emb_dim = 9

# def get_xbox_model(infile, outfile):
#     fout = open(outfile, 'w')
#     with open(infile, 'r') as fin:
#         for line in fin:
#             out_list = []
#             feasign, emb = re.split('\t', line.strip('\n'))
#             emb = re.split(' ', emb)
#             out_list.append(feasign)
#             out_list.extend(['0'] * 5)
#             out_list.append(emb[0])
#             out_list.append('0')
#             out_list.extend(emb[1:])
#             out_list.extend(['0'] * len(emb[1:]))
#             fout.write('{}\n'.format(' '.join(out_list)))

#     fout.close()


def get_xbox_model(infile, outfile):
    fout = open(outfile, 'w')
    out_list = ['0', '1', '0', '1']
    out_str1 = '\t'.join(out_list)
    out_list = ['0.000000'] * (3 * emb_dim + 3)
    out_str2 = ','.join(out_list)
    fout.write('{}\t{}\n'.format(out_str1, out_str2))
    with open(infile, 'r') as fin:
        for line in fin:
            out_list = []
            feasign, emb = re.split('\t', line.strip('\n'))
            emb = re.split(',', emb.strip(','))
            if len(emb) < emb_dim:
                emb.extend(['0.000000'] * (emb_dim - len(emb)))
            out_list.append(feasign)
            out_list.extend(['1', '0', '1'])
            out_str1 = '\t'.join(out_list)
            out_list = []
            out_list.extend(emb)
            out_list.extend(['0.000000'] * emb_dim)
            out_list.extend(['0.000000'] * emb_dim)
            out_list.extend(['0.000000'] * 3)
            out_str2 = ','.join(out_list)
            fout.write('{}\t{}\n'.format(out_str1, out_str2))
    fout.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='输入文件，cube结果', dest='infile', required=True)
    parser.add_argument(
        '-o', help='输出文件，xbox模型', dest='outfile', required=True)
    args = parser.parse_args()
    get_xbox_model(args.infile, args.outfile)
