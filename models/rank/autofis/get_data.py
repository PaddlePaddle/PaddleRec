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

import numpy as np
import os
from Criteo import Criteo
from tqdm import tqdm

batch_size = 1024

train_data_param = {
    'gen_type': 'train',
    'random_sample': True,
    'batch_size': batch_size,
    'split_fields': False,
    'on_disk': True,
    'squeeze_output': True,
}
test_data_param = {
    'gen_type': 'test',
    'random_sample': False,
    'batch_size': batch_size,
    'split_fields': False,
    'on_disk': True,
    'squeeze_output': True,
}

dataset = Criteo(initialized=True)
train_gen = dataset.batch_generator(train_data_param)
test_gen = dataset.batch_generator(test_data_param)

output_dir = 'data/whole_data'
xs = []
ys = []
for x, y in tqdm(train_gen):
    xs.append(x)
    ys.append(y)

x = np.concatenate(xs, 0)
y = np.concatenate(ys, 0)
print(x.shape)
np.save(os.path.join(output_dir, 'train', 'train_x.npy'), x)
np.save(os.path.join(output_dir, 'train', 'train_y.npy'), y)

xs = []
ys = []
for x, y in tqdm(test_gen):
    xs.append(x)
    ys.append(y)

x = np.concatenate(xs, 0)
y = np.concatenate(ys, 0)
print(x.shape)
np.save(os.path.join(output_dir, 'test', 'test_x.npy'), x)
np.save(os.path.join(output_dir, 'test', 'test_y.npy'), y)
