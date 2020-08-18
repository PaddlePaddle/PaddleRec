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

import paddle
import numpy as np

# Build a random data set.
sample_size = 100
batch_size = 32
watch_vec_size = 64
search_vec_size = 64
other_feat_size = 64
output_size = 100

watch_vecs = np.random.rand(batch_size * sample_size, watch_vec_size).tolist()
search_vecs = np.random.rand(batch_size * sample_size,
                             search_vec_size).tolist()
other_vecs = np.random.rand(batch_size * sample_size, other_feat_size).tolist()
labels = np.random.randint(
    output_size, size=(batch_size * sample_size)).tolist()

output_path = "./data/train/data.txt"
with open(output_path, 'w') as fout:
    for i in range(batch_size * sample_size):
        _str_ = ','.join(map(str, watch_vecs[i])) + ";" + ','.join(
            map(str, search_vecs[i])) + ";" + ','.join(
                map(str, other_vecs[i])) + ";" + str(labels[i])
        fout.write(_str_)
        fout.write("\n")
