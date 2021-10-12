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

import paddle
import paddle.distributed.fleet as fleet
import os
import sys

from confluent_kafka import Consumer


class KFKConsumer(Consumer):
    def __init__(self, hosts, gid, subs=[], timeout=5):
        config = {
            'bootstrap.servers': hosts,
            'group.id': gid,
            'message.max.bytes': 30000000,
            'default.topic.config': {
                'auto.offset.reset': 'earliest'
            }
        }
        Consumer.__init__(self, config)
        if subs:
            self.subscribe(subs)
        self.timeout = timeout

    def __iter__(self):
        return self

    def __next__(self):
        try:
            msg = self.poll(self.timeout)
            if msg.error() or msg.value() == 'Broker: No more messages':
                return None
            return msg.value().decode('utf-8')
        except:
            return None


cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)


class WideDeepDatasetReader(fleet.MultiSlotDataGenerator):
    def line_process(self, line):
        features = line.rstrip('\n').split('\t')
        dense_feature = []
        sparse_feature = []
        for idx in continuous_range_:
            if features[idx] == "":
                dense_feature.append(0.0)
            else:
                dense_feature.append(
                    (float(features[idx]) - cont_min_[idx - 1]) /
                    cont_diff_[idx - 1])
        for idx in categorical_range_:
            sparse_feature.append([hash(str(idx) + features[idx]) % hash_dim_])
        label = [int(features[0])]
        return [dense_feature] + sparse_feature + [label]

    def generate_sample(self, line):
        def wd_reader():
            for line in kfkc:
                if line is None:
                    continue
                input_data = self.line_process(line)
                feature_name = ["dense_input"]
                for idx in categorical_range_:
                    feature_name.append("C" + str(idx - 13))
                feature_name.append("label")
                yield zip(feature_name, input_data)

        return wd_reader


if __name__ == "__main__":
    my_data_generator = WideDeepDatasetReader()
    #my_data_generator.set_batch(16)

    hosts = os.getenv('KAFKA_HOSTS')
    gid = os.getenv('KAFKA_GID')
    subs = os.getenv('KAFKA_TOPICS').split(',')
    kfkc = KFKConsumer(hosts, gid, subs)

    my_data_generator.run_from_memory()
