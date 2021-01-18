#!/bin/bash
cd data

echo "---> Download movielens 1M data ..."
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
echo "---> Unzip ml-1m.zip ..."
unzip ml-1m.zip
rm ml-1m.zip

echo "---> Split movielens data ..."
python split.py

mkdir -p train/
mkdir -p test/

echo "---> Process train & test data ..."
python3 process_ml_1m.py process_raw ./ml-1m/train.dat | sort -t $'\t' -k 9 -n > log.data.train
python3 process_ml_1m.py process_raw ./ml-1m/test.dat | sort -t $'\t' -k 9 -n > log.data.test
python3 process_ml_1m.py hash log.data.train > data.txt
python3 padding.py ./data.txt > ./train/data.txt
python3 process_ml_1m.py hash log.data.test > data.txt
python3 padding.py ./data.txt > ./test/data.txt

rm data.txt
rm log.data.train
rm log.data.test


cd ..


echo "---> Finish data process"
