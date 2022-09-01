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

import math
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='criteo dataset scale')
parser.add_argument(
    '--src_path',
    type=str,
    required=False,
    default='./Criteo',
    help='source path')
args = parser.parse_args()


def scale(x):
    if x > 2:
        x = int(math.log(float(x))**2)
    return x


def scale_each_fold():
    for i in range(1, 11):
        print('now part %d' % i)
        data = np.load(
            os.path.join(args.src_path, 'part' + str(i), 'train_x.npy'),
            allow_pickle=True)
        part = data[:, 0:13]
        for j in range(part.shape[0]):
            if j % 100000 == 0:
                print(j)
            part[j] = list(map(scale, part[j]))
        np.save(
            os.path.join(args.src_path, 'part' + str(i), 'train_x2.npy'), data)


if __name__ == '__main__':
    scale_each_fold()
