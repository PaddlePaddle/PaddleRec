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

from confluent_kafka import Producer


class KFKProducer(Producer):
    def __init__(self, hosts, topic=None):
        config = {'bootstrap.servers': hosts, 'message.max.bytes': 30000000}
        Producer.__init__(self, config)
        self.topic = topic

    def send(self, message, topic=None):
        try:
            self.poll(0)
            pt = topic or self.topic
            self.produce(pt, message)
            return True
        except Exception as e:
            print("KFK to [{}] with [{}] failed : {}".format((
                topic or self.topic), message[:1000], e))
            return False


if __name__ == '__main__':

    filename = ''
    hosts = ""
    tp = "wide-and-deep-data"

    kfk = KFKProducer(hosts, topic=tp)

    with open(filename, 'r') as f:
        for line in f:
            kfk.send(line)

    kfk.flush()
    print("produce done")
