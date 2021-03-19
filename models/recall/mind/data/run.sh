
wget https://cloud.tsinghua.edu.cn/f/e5c4211255bc40cba828/?dl=1

tar -xvf data.tar.gz

rm -rf train valid
mkdir train
mkdir valid

mv data/book_data/book_train.txt  train
python preprocess.py -type valid -maxlen 20
rm -rf data.tar.gz
rm -rf data
