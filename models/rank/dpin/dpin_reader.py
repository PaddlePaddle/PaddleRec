#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import numpy as np
from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.config = config

        self.K = self.config.get("hyper_parameters.K", 3)  # Position的个数
        self.L = self.config.get("hyper_parameters.L", 10)  # 同一位置考虑历史记录
        self.batch_size = self.config.get("runner.train_batch_size", 32)

    def __iter__(self):
        for file in self.file_list:
            with open(file, "r") as f:
                for line in f:
                    line = line.strip().split(';')
                    # 至少得有7项：历史点击商品序列;历史点击商品类别序列;历史点击商品位置;推荐广告商品序列;推荐广告商品类别;推荐广告商品位置;点击标记
                    if len(line) < 7:
                        continue
                    ## history item [undefined]
                    hist_item = line[0].split()
                    hist_item = [int(x) for x in hist_item]
                    hist_item = np.array(hist_item)
                    ## history category [undefined]
                    hist_cat = line[1].split()
                    hist_cat = [int(x) for x in hist_cat]
                    hist_cat = np.array(hist_cat)
                    ## history position [undefined]
                    hist_pos = line[2].split()
                    hist_pos = [int(x) for x in hist_pos]
                    hist_pos = np.array(hist_pos)
                    ## !important, given the position, preprocess the item and category
                    hist_item_proc = []
                    hist_cate_proc = []
                    for i in range(1, self.K + 1):
                        tmp_item = hist_item[hist_pos == i].tolist()[:self.L]
                        tmp_item += [0] * (self.L - len(tmp_item))
                        hist_item_proc.append(tmp_item)
                        tmp_cate = hist_cat[hist_pos == i].tolist()[:self.L]
                        tmp_cate += [0] * (self.L - len(tmp_cate))
                        hist_cate_proc.append(tmp_cate)

                    ## target information
                    target_item = int(line[3])
                    target_cat = int(line[4])
                    target_pos = int(line[5])
                    label = float(line[6])
                    position = [i for i in range(self.K)]

                    res = []
                    res.append(
                        np.array(hist_item_proc).astype('int64'))  # [*, K, L]
                    res.append(
                        np.array(hist_cate_proc).astype('int64'))  # [*, K, L]
                    res.append(np.array(target_item).astype('int64'))  # [*]
                    res.append(np.array(target_cat).astype('int64'))  # [*]
                    res.append(np.array(target_pos).astype('int64'))  # [*]
                    res.append(np.array(position).astype('int64'))  # [*, K]
                    res.append(np.array(label).astype('float32'))  # [*]
                    yield res
