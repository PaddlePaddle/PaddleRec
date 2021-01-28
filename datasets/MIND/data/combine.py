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

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

if __name__ == '__main__':
    count = 0
    cate_dict = {"<unk>": 0}
    sub_cate_dict = {"<unk>": 0}
    id_map = {}
    list = ["train_raw/news.tsv", "dev_raw/news.tsv", "test_raw/news.tsv"]
    with open("news_backup.tsv", "w") as new_w:
        with open("kkk/temp.txt", "w") as w:
            for file in list:
                with open(file, "r") as f:
                    for l in f:
                        line = l.split("\t")
                        id = line[0]
                        if id in id_map:
                            continue
                        new_w.write(l)
                        id_map[id] = len(id_map)
                        cate = line[1]
                        sub_cate = line[2]
                        title = line[3]
                        content = line[4]
                        w.write(title + "\n")
                        w.write(content + "\n")
                        if cate not in cate_dict:
                            cate_dict[cate] = len(cate_dict)
                        if sub_cate not in sub_cate_dict:
                            sub_cate_dict[sub_cate] = len(sub_cate_dict)

    with open("cate_map", "w") as w1:
        for key in cate_dict:
            w1.write(key + "\t" + str(cate_dict[key]) + "\n")
    with open("sub_cate_map", "w") as w2:
        for key in sub_cate_dict:
            w2.write(key + "\t" + str(sub_cate_dict[key]) + "\n")

    #print(count)

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
