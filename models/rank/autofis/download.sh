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

mkdir -p data/whole_data
mkdir tmp
wget https://paddlerec.bj.bcebos.com/datasets/criteo_8d/part0.tar -O tmp/part0.tar
wget https://paddlerec.bj.bcebos.com/datasets/criteo_8d/part1.tar -O tmp/part1.tar
wget https://paddlerec.bj.bcebos.com/datasets/criteo_8d/part2.tar -O tmp/part2.tar
wget https://paddlerec.bj.bcebos.com/datasets/criteo_8d/part3.tar -O tmp/part3.tar
wget https://paddlerec.bj.bcebos.com/datasets/criteo_8d/part4.tar -O tmp/part4.tar
wget https://paddlerec.bj.bcebos.com/datasets/criteo_8d/part5.tar -O tmp/part5.tar
cat tmp/part* > tmp/criteo_8d.tar
tar xvf tmp/criteo_8d.tar -C tmp

python get_data.py tmp

rm -r tmp
