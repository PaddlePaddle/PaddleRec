#Copyright (c) 2018 Chence Shi

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import argparse
import os

parser = argparse.ArgumentParser(description='criteo dataset preprocesser')
parser.add_argument(
    '--source_data',
    type=str,
    required=True,
    default='./criteo.txt',
    help='source path')
parser.add_argument(
    '--output_path',
    type=str,
    required=True,
    default='./Criteo',
    help='output path')
args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

train_path = args.source_data
f1 = open(train_path, 'r')
dic = {}
# generate three fold.
# train_x: value
# train_i: index
# train_y: label
f_train_value = open(os.path.join(args.output_path, 'train_x.txt'), 'w')
f_train_index = open(os.path.join(args.output_path, 'train_i.txt'), 'w')
f_train_label = open(os.path.join(args.output_path, 'train_y.txt'), 'w')

for i in range(39):
    dic[i] = {}

cnt_train = 0

#for debug
#limits = 10000
index = [1] * 26
for line in f1:
    cnt_train += 1
    if cnt_train % 100000 == 0:
        print('now train cnt : %d\n' % cnt_train)
    #if cnt_train > limits:
    #	break
    split = line.strip('\n').split('\t')
    # 0-label, 1-13 numerical, 14-39 category
    for i in range(13, 39):
        #dic_len = len(dic[i])
        if split[i + 1] not in dic[i]:
            # [1, 0] 1 is the index for those whose appear times <= 10   0 indicates the appear times
            dic[i][split[i + 1]] = [1, 0]
        dic[i][split[i + 1]][1] += 1
        if dic[i][split[i + 1]][0] == 1 and dic[i][split[i + 1]][1] > 10:
            index[i - 13] += 1
            dic[i][split[i + 1]][0] = index[i - 13]
f1.close()
print('total entries :%d\n' % (cnt_train - 1))

# calculate number of category features of every dimension
kinds = [13]
for i in range(13, 39):
    kinds.append(index[i - 13])
print('number of dimensions : %d' % (len(kinds) - 1))
print(kinds)

for i in range(1, len(kinds)):
    kinds[i] += kinds[i - 1]
print(kinds)

# make new data

f1 = open(train_path, 'r')
cnt_train = 0
print('remake training data...\n')
for line in f1:
    cnt_train += 1
    if cnt_train % 100000 == 0:
        print('now train cnt : %d\n' % cnt_train)
    #if cnt_train > limits:
    #	break
    entry = ['0'] * 39
    index = [None] * 39
    split = line.strip('\n').split('\t')
    label = str(split[0])
    for i in range(13):
        if split[i + 1] != '':
            entry[i] = (split[i + 1])
        index[i] = (i + 1)
    for i in range(13, 39):
        if split[i + 1] != '':
            entry[i] = '1'
        index[i] = (dic[i][split[i + 1]][0])
    for j in range(26):
        index[13 + j] += kinds[j]
    index = [str(item) for item in index]
    f_train_value.write(' '.join(entry) + '\n')
    f_train_index.write(' '.join(index) + '\n')
    f_train_label.write(label + '\n')
f1.close()

f_train_value.close()
f_train_index.close()
f_train_label.close()
