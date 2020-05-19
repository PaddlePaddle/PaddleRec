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
echo "begin to download data"

cd raw_data && python download.py
mkdir diginetica
python preprocess.py --dataset diginetica

echo "begin to convert data (binary -> txt)"
python convert_data.py --data_dir diginetica

cat diginetica/train.txt | wc -l >> diginetica/config.txt

mkdir train_data
mv diginetica/train.txt train_data

mkdir test_data
mv diginetica/test.txt test_data


