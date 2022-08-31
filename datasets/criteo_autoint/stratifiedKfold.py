#Email of the author: zjduan@pku.edu.cn
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import argparse 


parser = argparse.ArgumentParser(description='criteo dataset k-fold')
parser.add_argument('--src_path', type=str, required=False, default='./Criteo', help='source path')
parser.add_argument('--num_split', type=int, required=False, default=10, help='number of K-fold split')
parser.add_argument('--random_seed', type=int, required=False, default=2018, help='random seed')
args = parser.parse_args()


scale = ""
train_x_name = "train_x.npy"
train_y_name = "train_y.npy"


def _load_data(_nrows=None, debug = False):

    train_x = pd.read_csv(os.path.join(args.src_path, 'train_x.txt'),header=None,sep=' ',nrows=_nrows, dtype=np.float)
    train_y = pd.read_csv(os.path.join(args.src_path, 'train_y.txt'),header=None,sep=' ',nrows=_nrows, dtype=np.int32)

    train_x = train_x.values
    train_y = train_y.values.reshape([-1])
    

    print('data loading done!')
    print('training data : %d' % train_y.shape[0])
    
    assert train_x.shape[0]==train_y.shape[0]

    return train_x, train_y


def save_x_y(fold_index, train_x, train_y):
    _get = lambda x, l: [x[i] for i in l]
    for i in range(len(fold_index)):
        print("now part %d" % (i+1))
        part_index = fold_index[i]
        Xv_train_, y_train_ = _get(train_x, part_index), _get(train_y, part_index)
        save_dir_Xv = os.path.join(args.src_path, 'part{}'.format(i+1))
        save_dir_y = os.path.join(args.src_path, 'part{}'.format(i+1))
        if (os.path.exists(save_dir_Xv) == False):
            os.makedirs(save_dir_Xv)
        if (os.path.exists(save_dir_y) == False):
            os.makedirs(save_dir_y)
        save_path_Xv  = os.path.join(save_dir_Xv, train_x_name)
        save_path_y  = os.path.join(save_dir_y, train_y_name)
        np.save(save_path_Xv, Xv_train_)
        np.save(save_path_y, y_train_)


def save_i(fold_index):
    _get = lambda x, l: [x[i] for i in l]
    train_i = pd.read_csv(os.path.join(args.src_path, 'train_i.txt'), header=None,sep=' ',nrows=None, dtype=np.int32) 
    train_i = train_i.values 
    feature_size = train_i.max() + 1
    print ("feature_size = %d" % feature_size) 
    feature_size = [feature_size]
    feature_size = np.array(feature_size)
    np.save(os.path.join(args.src_path, "feature_size.npy"), feature_size)

    print("train_i size: %d" % len(train_i))

    for i in range(len(fold_index)):
        print("now part %d" % (i+1))
        part_index = fold_index[i]
        Xi_train_ = _get(train_i, part_index)
        save_path_Xi  = os.path.join(args.src_path, "part" + str(i+1), 'train_i.npy')
        np.save(save_path_Xi, Xi_train_)


def main():
    train_x, train_y = _load_data()
    print('loading data done!')

    folds = list(StratifiedKFold(n_splits=10, shuffle=True,
                             random_state=args.random_seed).split(train_x, train_y))

    fold_index = []
    for i,(train_id, valid_id) in enumerate(folds):
        fold_index.append(valid_id)

    print("fold num: %d" % (len(fold_index)))

    fold_index = np.array(fold_index)
    np.save(os.path.join(args.src_path,'fold_index.npy'), fold_index, allow_pickle=True)

    save_x_y(fold_index, train_x, train_y)
    print("save train_x_y done!")

    fold_index = np.load(os.path.join(args.src_path, "fold_index.npy"), allow_pickle=True)
    save_i(fold_index)
    print("save index done!")

if __name__ == "__main__":
    main()
