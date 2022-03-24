# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from math import sqrt
import sys
import os


#element multiple and sum
def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab


#calculate relation of 2 in pearson
def relation_cal_2(x, y):
    n = len(x)
    #sum
    sum1 = sum(x)
    sum2 = sum(y)
    #element multipl and sum
    sumofxy = multipl(x, y)
    #square sum
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    #pearson
    den = sqrt((sumofx2 - float(sum1**2) / n) * (sumofy2 - float(sum2**2) / n))
    return num / den


#calculate tasks relation in pearson
def relation_cal(file_list):
    file_num = len(file_list)
    result_sum = 0
    pair_sum = 0
    for i in range(file_num):
        for j in range(i + 1, file_num):
            result_sum += relation_cal_2(file_list[i], file_list[j])
            pair_sum += 1

    return float(result_sum / pair_sum)


#read the label data
def data_read(file_path):
    dirs = os.listdir(file_path)
    file_list = []
    for file in dirs:
        f_l = []
        dir = os.path.join(file_path, file)
        with open(dir, 'r') as f:
            for line in f:
                label = line.strip()
                f_l.append(float(label))
        f.close()
        file_list.append(f_l)
    return file_list


if __name__ == "__main__":
    file_path = sys.argv[1]
    file_list = data_read(file_path)
    co_ralation = relation_cal(file_list)
    print("The tasks' relation is: ", co_ralation)
