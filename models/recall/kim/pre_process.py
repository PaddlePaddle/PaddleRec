import os
import numpy as np

data_root_path = './data/train'
output_data_path = './data/dev'

with open(os.path.join(data_root_path, 'train', 'news.tsv')) as f:
    news1 = f.readlines()
with open(os.path.join(data_root_path, 'dev', 'news.tsv')) as f:
    news2 = f.readlines()

news = []
news_dict = {}
for l in news1 + news2:
    nid = l.strip('\n').split('\t')[0]
    if not nid in news_dict:
        news_dict[nid] = 1
        news.append(l)

with open(os.path.join(output_data_path, 'docs.tsv'), 'w') as f:
    for i in range(len(news)):
        f.write(news[i])

with open(os.path.join(data_root_path, 'train', 'behaviors.tsv')) as f:
    behaviors1 = f.readlines()
with open(os.path.join(data_root_path, 'dev', 'behaviors.tsv')) as f:
    behaviors2 = f.readlines()

train_behaviors = []
val_behaviors = []
index = np.random.permutation(len(behaviors1))
num = int(0.95 * len(behaviors1))
for i in range(num):
    train_behaviors.append(behaviors1[i])
for i in range(num, len(behaviors1)):
    val_behaviors.append(behaviors1[i])
test_behaviors = behaviors2

with open(os.path.join(output_data_path, 'train.tsv'), 'w') as f:
    for i in range(len(train_behaviors)):
        f.write(train_behaviors[i])
with open(os.path.join(output_data_path, 'val.tsv'), 'w') as f:
    for i in range(len(val_behaviors)):
        f.write(val_behaviors[i])
with open(os.path.join(output_data_path, 'test.tsv'), 'w') as f:
    for i in range(len(test_behaviors)):
        f.write(test_behaviors[i])
