
wget https://paddlerec.bj.bcebos.com/datasets/AmazonBook/AmazonBook.tar.gz

tar -xvf AmazonBook.tar.gz

rm -rf train valid
mkdir train
mkdir valid

mv book_data/book_train.txt  train
python preprocess.py -type valid -maxlen 20
