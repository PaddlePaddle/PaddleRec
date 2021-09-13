# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import pickle
import pandas as pd


def to_df(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df


print("start to analyse reviews_Electronics_5.json")
reviews_df = to_df('./raw_data/reviews_Electronics_5.json')
with open('./raw_data/reviews.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

print("start to analyse meta_Electronics.json")
meta_df = to_df('./raw_data/meta_Electronics.json')
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
with open('./raw_data/meta.pkl', 'wb') as f:
    pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)
