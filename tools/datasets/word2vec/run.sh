#! /bin/bash

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

mkdir -p data/all_dict
mkdir -p data/all_train
mkdir -p data/all_test

# download train_data
mkdir raw_data
wget --no-check-certificate https://paddlerec.bj.bcebos.com/word2vec/1-billion-word-language-modeling-benchmark-r13output.tar
tar xvf 1-billion-word-language-modeling-benchmark-r13output.tar
mv 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ raw_data/

# download test data
wget --no-check-certificate https://paddlerec.bj.bcebos.com/word2vec/test_dir.tar
tar xzvf test_dir.tar -C raw_data
mv raw_data/data/test_dir/* data/all_test/

# preprocess data
python preprocess.py --build_dict --build_dict_corpus_dir raw_data/training-monolingual.tokenized.shuffled --dict_path raw_data/word_count_dict.txt
python preprocess.py --filter_corpus --dict_path raw_data/word_count_dict.txt --input_corpus_dir raw_data/training-monolingual.tokenized.shuffled --output_corpus_dir raw_data/convert_text8 --min_count 5 --downsample 0.001
python preprocess.py --data_resplit --input_corpus_dir=raw_data/convert_text8 --output_corpus_dir=data/all_train

mv raw_data/word_count_dict.txt data/all_dict/
mv raw_data/word_count_dict.txt_word_to_id_ data/all_dict/word_id_dict.txt
rm -rf raw_data
