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


def count(file):
    count = 0
    with open(file, "r") as r:
        for l in r:
            count = count + 1
    return count


with open("dict/yaml_info.txt", "w") as w:
    word_count = count("dict/word_dict")
    cate = count("dict/cate_map")
    sub_cate = count("dict/sub_cate_map")
    w.write("word_dict_size: " + str(word_count) + "\n")
    w.write("category_size: " + str(cate) + "\n")
    w.write("sub_category_size " + str(sub_cate) + "\n")
