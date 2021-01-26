#!/bin/bash

echo "...........load  data................."
wget --no-check-certificate 'https://paddlerec.bj.bcebos.com/match_pyramid/match_pyramid_data.tar.gz'
tar -xvf ./match_pyramid_data.tar.gz
mkdir ./big_train
mkdir ./big_test
echo "...........data process..............."
python ./process.py
