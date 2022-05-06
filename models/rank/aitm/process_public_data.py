'''
process the Ali-CCP (Alibaba Click and Conversion Prediction) dataset.
https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408

@The author:
Dongbo Xi (xidongbo@meituan.com)
'''
import numpy as np
import joblib
import re
import random
random.seed(2020)
np.random.seed(2020)
data_path = 'data/sample_skeleton_{}.csv'
common_feat_path = 'data/common_features_{}.csv'
enum_path = 'data/ctrcvr_enum.pkl'
write_path = 'data/ctr_cvr'
use_columns = [
    '101',
    '121',
    '122',
    '124',
    '125',
    '126',
    '127',
    '128',
    '129',
    '205',
    '206',
    '207',
    '216',
    '508',
    '509',
    '702',
    '853',
    '301']

class process(object):
    def __init__(self):
        pass

    def process_train(self):
        c = 0
        common_feat_dict = {}
        with open(common_feat_path.format('train'), 'r') as fr:
            for line in fr:
                line_list = line.strip().split(',')
                kv = np.array(re.split('\x01|\x02|\x03', line_list[2]))
                key = kv[range(0, len(kv), 3)]
                value = kv[range(1, len(kv), 3)]
                feat_dict = dict(zip(key, value))
                common_feat_dict[line_list[0]] = feat_dict
                c += 1
                if c % 100000 == 0:
                    print(c)
        print('join feats...')
        c = 0
        vocabulary = dict(zip(use_columns, [{}  for _ in range(len(use_columns))]))
        with open(data_path.format('train') + '.tmp', 'w') as fw:
            fw.write('click,purchase,' + ','.join(use_columns) + '\n')
            with open(data_path.format('train'), 'r') as fr:
                for line in fr:
                    line_list = line.strip().split(',')
                    if line_list[1] == '0' and line_list[2] == '1':
                        continue
                    kv = np.array(re.split('\x01|\x02|\x03', line_list[5]))
                    key = kv[range(0, len(kv), 3)]
                    value = kv[range(1, len(kv), 3)]
                    feat_dict = dict(zip(key, value))
                    feat_dict.update(common_feat_dict[line_list[3]])
                    feats = line_list[1:3]
                    for k in use_columns:
                        feats.append(feat_dict.get(k, '0'))
                    fw.write(','.join(feats) + '\n')
                    for k, v in feat_dict.items():
                        if k in use_columns:
                            if v in vocabulary[k]:
                                vocabulary[k][v] += 1
                            else:
                                vocabulary[k][v] = 0
                    c += 1
                    if c % 100000 == 0:
                        print(c)
        print('before filter low freq:')
        for k, v in vocabulary.items():
            print(k + ':' + str(len(v)))
        new_vocabulary = dict(
            zip(use_columns, [set() for _ in range(len(use_columns))]))
        for k, v in vocabulary.items():
            for k1, v1 in v.items():
                if v1 > 10:
                    new_vocabulary[k].add(k1)
        vocabulary = new_vocabulary
        print('after filter low freq:')
        for k, v in vocabulary.items():
            print(k + ':' + str(len(v)))
        joblib.dump(vocabulary, enum_path, compress=3)

        print('encode feats...')
        vocabulary = joblib.load(enum_path)
        feat_map = {}
        for feat in use_columns:
            feat_map[feat] = dict(
                zip(vocabulary[feat], range(1, len(vocabulary[feat]) + 1)))
        c = 0
        with open(write_path + '.train', 'w') as fw1:
            with open(write_path + '.dev', 'w') as fw2:
                fw1.write('click,purchase,' + ','.join(use_columns) + '\n')
                fw2.write('click,purchase,' + ','.join(use_columns) + '\n')
                with open(data_path.format('train') + '.tmp', 'r') as fr:
                    fr.readline()  # remove header
                    for line in fr:
                        line_list = line.strip().split(',')
                        new_line = line_list[:2]
                        for value, feat in zip(line_list[2:], use_columns):
                            new_line.append(
                                str(feat_map[feat].get(value, '0')))
                        if random.random() >= 0.9:
                            fw2.write(','.join(new_line) + '\n')
                        else:
                            fw1.write(','.join(new_line) + '\n')
                        c += 1
                        if c % 100000 == 0:
                            print(c)

    def process_test(self):
        c = 0
        common_feat_dict = {}
        with open(common_feat_path.format('test'), 'r') as fr:
            for line in fr:
                line_list = line.strip().split(',')
                kv = np.array(re.split('\x01|\x02|\x03', line_list[2]))
                key = kv[range(0, len(kv), 3)]
                value = kv[range(1, len(kv), 3)]
                feat_dict = dict(zip(key, value))
                common_feat_dict[line_list[0]] = feat_dict
                c += 1
                if c % 100000 == 0:
                    print(c)
        print('join feats...')
        c = 0
        with open(data_path.format('test') + '.tmp', 'w') as fw:
            fw.write('click,purchase,' + ','.join(use_columns) + '\n')
            with open(data_path.format('test'), 'r') as fr:
                for line in fr:
                    line_list = line.strip().split(',')
                    if line_list[1] == '0' and line_list[2] == '1':
                        continue
                    kv = np.array(re.split('\x01|\x02|\x03', line_list[5]))
                    key = kv[range(0, len(kv), 3)]
                    value = kv[range(1, len(kv), 3)]
                    feat_dict = dict(zip(key, value))
                    feat_dict.update(common_feat_dict[line_list[3]])
                    feats = line_list[1:3]
                    for k in use_columns:
                        feats.append(str(feat_dict.get(k, '0')))
                    fw.write(','.join(feats) + '\n')
                    c += 1
                    if c % 100000 == 0:
                        print(c)

        print('encode feats...')
        vocabulary = joblib.load(enum_path)
        feat_map = {}
        for feat in use_columns:
            feat_map[feat] = dict(
                zip(vocabulary[feat], range(1, len(vocabulary[feat]) + 1)))
        c = 0
        with open(write_path + '.test', 'w') as fw:
            fw.write('click,purchase,' + ','.join(use_columns) + '\n')
            with open(data_path.format('test') + '.tmp', 'r') as fr:
                fr.readline()  # remove header
                for line in fr:
                    line_list = line.strip().split(',')
                    new_line = line_list[:2]
                    for value, feat in zip(line_list[2:], use_columns):
                        new_line.append(str(feat_map[feat].get(value, '0')))
                    fw.write(','.join(new_line) + '\n')
                    c += 1
                    if c % 100000 == 0:
                        print(c)


if __name__ == "__main__":
    pros = process()
    pros.process_train()
    pros.process_test()