#!/bin/bash
FILENAME=$1
# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer']
MODE=$2

# prepare pretrained weights and dataset 
wget -nc -P  ./tests/save_wide_deep_model https://paddlerec.bj.bcebos.com/wide_deep/wide_deep.tar
cd tests/save_wide_deep_model && tar -xvf wide_deep.tar && rm -rf wide_deep.tar && cd ../../

mkdir -p ./tests/data/train
mkdir -p ./tests/data/infer
if [ ${MODE} = "lite_train_infer" ];then
    cp -r ./models/rank/wide_deep/data/sample_data/train/* ./tests/data/train
    cp -r ./models/rank/wide_deep/data/sample_data/train/* ./tests/data/infer
    echo "demo data ready"
elif [ ${MODE} = "whole_train_infer" ];then
    cd ./datasets/criteo
    bash run.sh
    cd ../..
    cp -r ./datasets/criteo/slot_train_data_full/* ./tests/data/train
    cp -r ./datasets/criteo/slot_test_data_full/* ./tests/data/infer
    echo "whole data ready"
elif [ ${MODE} = "whole_infer" ];then
    cd ./datasets/criteo
    bash run.sh
    cd ../..
    cp -r ./models/rank/wide_deep/data/sample_data/train/* ./tests/data/train
    cp -r ./datasets/criteo/slot_test_data_full/* ./tests/data/infer
else
    cd ./datasets/criteo
    bash run.sh
    cd ../..
    cp -r ./models/rank/wide_deep/data/sample_data/train/* ./tests/data/train
    cp -r ./datasets/criteo/slot_test_data_full/* ./tests/data/infer
fi
