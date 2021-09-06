# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from odps import ODPS
from odps.models import Schema, Column, Partition

import paddle
import paddle.distributed.fleet as fleet
import os
import sys

cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)


class WideDeepDatasetReader(fleet.MultiSlotDataGenerator):
    def line_process(self, record):
        label = [record.label]
        dense_feature = []
        sparse_feature = []
        for idx in continuous_range_:
            dense_feature.append((float(record[idx]) - cont_min_[idx - 1]) /
                                 cont_diff_[idx - 1])
        for idx in categorical_range_:
            sparse_feature.append([hash(str(idx) + record[idx]) % hash_dim_])
        return [dense_feature] + sparse_feature + [label]

    def generate_sample(self, line):
        def wd_reader():
            for record in table_reader:
                input_data = self.line_process(record)
                feature_name = ["dense_input"]
                for idx in categorical_range_:
                    feature_name.append("C" + str(idx - 13))
                feature_name.append("label")
                yield zip(feature_name, input_data)

        return wd_reader


if __name__ == "__main__":
    my_data_generator = WideDeepDatasetReader()
    #my_data_generator.set_batch(16)

    from config import *
    # config should include flowing configuration
    # access_id
    # secret_key
    # project
    # endpoint

    o = ODPS(access_id, secret_key, project, endpoint=endpoint)

    table_name = 'wide_and_deep'

    table = o.get_table(table_name)  #.to_df()

    table_reader = table.open_reader()

    my_data_generator.run_from_memory()
