# encoding=utf-8
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
import sys


def build_word_dict():
    word_file = "word_dict.txt"
    f = open(word_file, "r")
    lines = f.readlines()
    word_list_ids = range(1, len(lines) + 1)
    word_dict = dict(zip([word.strip() for word in lines], word_list_ids))
    f.close()
    return word_dict


def build_token_data(word_dict, txt_file, token_file):
    max_text_size = 100

    f = open(txt_file, "r")
    fout = open(token_file, "w")
    lines = f.readlines()
    i = 0

    for line in lines:
        line = line.strip("\n").split("\t")
        text = line[0].strip("\n").split(" ")
        tokens = []
        label = line[1]
        for word in text:
            if word in word_dict:
                tokens.append(str(word_dict[word]))
            else:
                tokens.append("0")

        seg_len = len(tokens)
        if seg_len < 5:
            continue
        if seg_len >= max_text_size:
            tokens = tokens[:max_text_size]
            seg_len = max_text_size
        else:
            tokens = tokens + ["0"] * (max_text_size - seg_len)
        text_tokens = " ".join(tokens)
        fout.write(text_tokens + " " + str(seg_len) + " " + label + "\n")
        if (i + 1) % 100 == 0:
            print(str(i + 1) + " lines OK")
        i += 1

    fout.close()
    f.close()


word_dict = build_word_dict()

txt_file = "test.tsv"
token_file = "test.txt"
build_token_data(word_dict, txt_file, token_file)

txt_file = "dev.tsv"
token_file = "dev.txt"
build_token_data(word_dict, txt_file, token_file)

txt_file = "train.tsv"
token_file = "train.txt"
build_token_data(word_dict, txt_file, token_file)
