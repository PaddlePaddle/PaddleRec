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

import sys
import codecs


def convert_format(input, output):
    with codecs.open(input, "r", encoding='utf-8') as rf:
        with codecs.open(output, "w", encoding='utf-8') as wf:
            last_sess = -1
            sign = 1
            i = 0
            for l in rf:
                i = i + 1
                if i == 1:
                    continue
                if (i % 1000000 == 1):
                    print(i)
                tokens = l.strip().split()
                if (int(tokens[0]) != last_sess):
                    if (sign):
                        sign = 0
                        wf.write(tokens[1] + " ")
                    else:
                        wf.write("\n" + tokens[1] + " ")
                    last_sess = int(tokens[0])
                else:
                    wf.write(tokens[1] + " ")


input = "rsc15_train_tr.txt"
output = "rsc15_train_tr_paddle.txt"
input2 = "rsc15_test.txt"
output2 = "rsc15_test_paddle.txt"
convert_format(input, output)
convert_format(input2, output2)
