import math
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='criteo dataset scale')
parser.add_argument('--src_path', type=str, required=False, default='./Criteo', help='source path')
args = parser.parse_args()


def scale(x):
    if x > 2:
        x = int(math.log(float(x))**2)
    return x



def scale_each_fold():
    for i in range(1,11):
        print('now part %d' % i)
        data = np.load(os.path.join(args.src_path, 'part'+str(i), 'train_x.npy'), allow_pickle=True)
        part = data[:,0:13]
        for j in range(part.shape[0]):
            if j % 100000 ==0:
                print(j)
            part[j] = list(map(scale, part[j]))
        np.save(os.path.join(args.src_path, 'part' + str(i), 'train_x2.npy'), data)   


if __name__ == '__main__':
    scale_each_fold()