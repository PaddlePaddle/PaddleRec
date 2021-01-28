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

file_base = ["train", "dev", "test"]
files = ["behaviors.tsv", "behaviors_dev.tsv", "behaviors_test.tsv"]
for base, aim in zip(files, file_base):
    with open(aim, "w") as w1:
        with open(base, "r") as f:
            for l in f:
                line = l.split("\t")
                visit = line[3]
                if len(visit) == 0:
                    continue
                sample = line[4].split(" ")
                pos_sample = ""
                neg_sample = ""
                for s in sample:
                    if len(s) > 0 and (s[-2:] == "-1" or s[-2:] == "-0"):
                        id = s.split("-")[0]
                        if id in article_map:
                            if s[-2:] == "-1":
                                if len(pos_sample) > 0:
                                    pos_sample = pos_sample + " "
                                pos_sample = pos_sample + id
                            else:
                                if len(neg_sample) > 0:
                                    neg_sample = neg_sample + " "
                                neg_sample = neg_sample + id

                if len(pos_sample) == 0 or len(neg_sample) == 0:
                    continue

                line = visit + "\t" + pos_sample + "\t" + neg_sample + "\n"
                if random.randint(1, 10) == 3:
                    w2.write(line)
                else:
                    w1.write(line)
