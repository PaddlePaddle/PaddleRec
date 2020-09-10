# encoding=utf-8
import os
import sys


def build_word_dict():
    word_file = "word_dict.txt"
    f = open(word_file, "r")
    word_dict = {}
    lines = f.readlines()
    for line in lines:
        word = line.strip().split("\t")
        word_dict[word[0]] = word[1]
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
