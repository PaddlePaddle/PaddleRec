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
#!/bin/bash

wget https://paddlerec.bj.bcebos.com/dssm%2Fbq.tar.gz
tar xzf dssm%2Fbq.tar.gz
rm -f dssm%2Fbq.tar.gz
mv bq/train.txt ./raw_data.txt
python3 preprocess.py
mkdir big_train
mv train.txt ./big_train
mkdir big_test
mv test.txt ./big_test
