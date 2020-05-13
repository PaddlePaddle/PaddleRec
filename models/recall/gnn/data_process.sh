#! /bin/bash

set -e
echo "begin to download data"

cd raw_data && python download.py
mkdir diginetica
python preprocess.py --dataset diginetica

echo "begin to convert data (binary -> txt)"
python convert_data.py --data_dir diginetica

cat diginetica/train.txt | wc -l >> diginetica/config.txt

mkdir train_data
mv diginetica/train.txt train_data

mkdir test_data
mv diginetica/test.txt test_data


