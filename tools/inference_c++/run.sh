#!/bin/bash
work_path=$(dirname $(readlink -f $0))

mkdir -p build
cd build
rm -rf *

LIB_DIR=${work_path}/paddle_inference
INFER_NAME=inference
WITH_MKL=ON
WITH_GPU=ON
cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DWITH_GPU=${WITH_GPU} \
  -DINFER_NAME=${INFER_NAME}
make
./inference --data_dir=/home/PaddleRec/models/rank/wide_deep/data/sample_data/train --model_file=/home/PaddleRec/tools/inference_c++/wide_deep/rec_inference.pdmodel --params_file=/home/PaddleRec/tools/inference_c++/wide_deep/rec_inference.pdiparams --batch_size=5
