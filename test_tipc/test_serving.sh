#!/bin/bash

MODEL=$1

# 安装 serving 相关包
#pip install paddle-serving-client -i https://mirror.baidu.com/pypi/simple
#pip install paddle-serving-server -i https://mirror.baidu.com/pypi/simple
#pip install paddle-serving-server-gpu -i https://mirror.baidu.com/pypi/simple
#pip install paddle_serving_app -i https://mirror.baidu.com/pypi/simple

# step 1, train & infer model
cd ../models/rank/${MODEL} && python -u ../../../tools/static_trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml

# step 2, convert
python -m paddle_serving_client.convert --dirname ./output_model_${MODEL}/0 --model_filename rec_inference.pdmodel --params_filename rec_inference.pdiparams

# step 3, 启动 web 服务端
python ../../../tools/webserver.py cpu 9393

# step 4, 测试部署的服务 (另开终端测试）
# cd ../models/rank/${MODEL} && python -u ../../../tools/rec_client.py --client_config=serving_client/serving_client_conf.prototxt --connect=0.0.0.0:9393 --use_gpu=false --data_dir=data/sample_data/train/ --reader_file=criteo_reader.py --batchsize=5 --client_mode=web

echo "~ done"
