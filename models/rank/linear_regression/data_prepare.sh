cd data
# 1. download data
python download_preprocess.py 

# 2. split data
python split.py

# 3. 数据拼接
python preprocess.py process_raw ml-1m/train.dat raw_train
python preprocess.py process_raw ml-1m/test.dat raw_test

# 4. hash
python preprocess.py hash raw_train > train_data/data
python preprocess.py hash raw_test > test_data/data
cd ..
