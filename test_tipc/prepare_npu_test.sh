#!/bin/bash

BASEDIR=$(dirname "$0")

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

REPO_ROOT_PATH=$(readlinkf ${BASEDIR}/../)

config_files=$(find ${REPO_ROOT_PATH}/test_tipc/configs -name "train_infer_python.txt")
for file in ${config_files}; do
   echo $file
   sed -i "s/runner.use_gpu/runner.use_npu/g" $file
   sed -i '30s/$/ runner.use_gpu=False/' $file
   sed -i '40s/$/ --use_gpu=false/' $file
   sed -i "s/--use_gpu:/--use_npu:/g" $file
done

yaml_files=$(find ${REPO_ROOT_PATH}/models/ -name "*.yaml")
for file in ${yaml_files}; do
   echo $file
   sed -i "s/use_gpu: True/use_gpu: False/g" $file
done

