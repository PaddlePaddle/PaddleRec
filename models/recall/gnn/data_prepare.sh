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

set -e

dataset=$1
src=$1

if [[ $src == "yoochoose1_4" || $src == "yoochoose1_64" ]];then
    src="yoochoose"
elif [[ $src == "diginetica" ]];then
    src="diginetica"
else
    echo "Usage: sh data_prepare.sh [diginetica|yoochoose1_4|yoochoose1_64]"
    exit 1
fi

echo "begin to download data"
cd data && python download.py $src
mkdir $dataset
python preprocess.py --dataset $src

echo "begin to convert data (binary -> txt)"
python convert_data.py --data_dir $dataset

cat ${dataset}/train.txt | wc -l >> config.txt

rm -rf train && mkdir train
mv ${dataset}/train.txt train

rm -rf test && mkdir test
mv ${dataset}/test.txt test
