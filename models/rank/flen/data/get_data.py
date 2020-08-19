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
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('./avazu_sample.txt')
data['day'] = data['hour'].apply(lambda x: str(x)[4:6])
data['hour'] = data['hour'].apply(lambda x: str(x)[6:])

sparse_features = [
    'hour',
    'C1',
    'banner_pos',
    'site_id',
    'site_domain',
    'site_category',
    'app_id',
    'app_domain',
    'app_category',
    'device_id',
    'device_model',
    'device_type',
    'device_conn_type',  # 'device_ip',
    'C14',
    'C15',
    'C16',
    'C17',
    'C18',
    'C19',
    'C20',
    'C21',
]

data[sparse_features] = data[sparse_features].fillna('-1', )

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

cols = [
    'click', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C1',
    'device_model', 'device_type', 'device_id', 'app_id', 'app_domain',
    'app_category', 'banner_pos', 'site_id', 'site_domain', 'site_category',
    'device_conn_type', 'hour'
]
# 计算每一个特征的最大值，作为vacob_size
data = data[cols]
line = ''
vacob_file = open('vacob_file.txt', 'w')
for col in cols[1:]:
    max_val = data[col].max()
    line += str(max_val) + ','
vacob_file.write(line)
vacob_file.close()

data.to_csv('./train_data/train_data.txt', index=False, header=None)
