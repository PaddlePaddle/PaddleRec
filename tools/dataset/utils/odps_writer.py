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

from config import *
# config should include flowing configuration
# access_id
# secret_key
# project
# endpoint
'''
For more information, ref to
https://pyodps.readthedocs.io/
'''

o = ODPS(access_id, secret_key, project, endpoint=endpoint)

label_col = [Column(name='label', type='bigint')]
dense_col = [
    Column(
        name='dense' + str(i), type='double') for i in range(1, 14)
]
sparse_col = [Column(name='C' + str(i), type='string') for i in range(14, 40)]

columns = label_col + dense_col + sparse_col

schema = Schema(
    columns=columns)  # schema = Schema(columns=columns, partitions=partitions)

table_name = 'wide_and_deep'

print(schema.columns)


def create_table():
    table = o.create_table(table_name, schema, if_not_exists=True)


#create_table()

table = o.get_table(table_name)  #.to_df()
print(table.to_df())


def write_data():
    records = []

    # prepare data
    input_file = './part-0'
    with open(input_file, 'r') as f:
        for line in f:
            example = []

            features = line.rstrip('\n').split('\t')
            label = int(features[0])
            example.append(label)

            for idx in range(1, 14):
                if features[idx] == "":
                    example.append(0.0)
                else:
                    example.append(float(features[idx]))
            for idx in range(14, 40):
                example.append(features[idx].encode('utf8'))

            records.append(example)

    with table.open_writer() as writer:
        writer.write(records)


#write_data()


def read_data():
    with table.open_reader() as reader:
        print("data count in table:", reader.count)
        for r in reader:
            print(r.label)
            #print([i for i in r[1:14]])
            print([i for i in r])
            break


read_data()
