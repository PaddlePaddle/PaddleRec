import argparse
import time
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default='sample',
    help='dataset dir: diginetica/yoochoose1_4/yoochoose1_64/sample')
opt = parser.parse_args()

def process_data(file_type):
    path = os.path.join(opt.data_dir, file_type)
    output_path = os.path.splitext(path)[0] + ".txt"
    data = pickle.load(open(path, 'rb'))
    data = list(zip(data[0], data[1]))
    length = len(data)
    with open(output_path, 'w') as fout:
        for i in range(length):
            fout.write(','.join(map(str, data[i][0])))
            fout.write('\t')
            fout.write(str(data[i][1]))
            fout.write("\n")

process_data("train")
process_data("test")

print('Done.')
