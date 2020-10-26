#!/bin/bash

echo "...........load  data................."
wget --no-check-certificate 'https://paddlerec.bj.bcebos.com/match_pyramid/match_pyramid_data.tar.gz'
mv ./match_pyramid_data.tar.gz ./data
rm -rf ./data/relation.test.fold1.txt 
tar -xvf ./data/match_pyramid_data.tar.gz
mkdir ./data/big_train
mkdir ./data/big_test
echo "...........data process..............."
python ./data/process.py
