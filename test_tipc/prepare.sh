#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer']
MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})
# The training params
model_name=$(func_parser_value "${lines[1]}")

# clear dataset
rm -rf ./test_tipc/data

if [ ${model_name} == "dnn" ]; then
    # prepare pretrained weights and dataset 
    wget -nc -P  ./test_tipc/save_dnn_model https://paddlerec.bj.bcebos.com/wide_deep/wide_deep.tar
    cd test_tipc/save_dnn_model && tar -xvf wide_deep.tar && rm -rf wide_deep.tar && cd ../../
    
    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/rank/dnn/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./models/rank/dnn/data/sample_data/train/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./datasets/criteo/slot_train_data_full/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./models/rank/dnn/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./models/rank/dnn/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
    elif [ ${MODE} = "benchmark_train" ];then
        cp -r ./models/rank/dnn/data/sample_data/train/* ./test_tipc/data/train
        echo "demo data ready"
    fi
 
elif [ ${model_name} == "wide_deep" ]; then
    # prepare pretrained weights and dataset 
    wget -nc -P  ./test_tipc/save_wide_deep_model https://paddlerec.bj.bcebos.com/wide_deep/wide_deep.tar
    cd test_tipc/save_wide_deep_model && tar -xvf wide_deep.tar && rm -rf wide_deep.tar && cd ../../
    
    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/rank/wide_deep/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./models/rank/wide_deep/data/sample_data/train/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./datasets/criteo/slot_train_data_full/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./models/rank/wide_deep/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./models/rank/wide_deep/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
    elif [ ${MODE} = "benchmark_train" ];then
        cp -r ./models/rank/wide_deep/data/sample_data/train/* ./test_tipc/data/train
        echo "demo data ready"
    fi
    
elif [ ${model_name} == "deepfm" ]; then
    # prepare pretrained weights and dataset 
    wget -nc -P  ./test_tipc/ https://paddlerec.bj.bcebos.com/deepfm/deepfm.tar
    cd test_tipc && tar -xvf deepfm.tar && rm -rf deepfm.tar && cd ..

    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/rank/deepfm/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./models/rank/deepfm/data/sample_data/train/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./datasets/criteo/slot_train_data_full/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./models/rank/deepfm/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./models/rank/deepfm/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
    fi

elif [ ${model_name} == "autoint" ]; then
    # prepare pretrained weights and dataset 

    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/rank/autoint/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./models/rank/autoint/data/sample_data/train/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/criteo_autoint
        bash run.sh
        cd ../..
        cp -r ./datasets/criteo_autoint/slot_train_data_full/* ./test_tipc/data/train
        cp -r ./datasets/criteo_autoint/slot_test_data_full/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/criteo_autoint
        bash run.sh
        cd ../..
        cp -r ./models/rank/autoint/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo_autoint/slot_test_data_full/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./models/rank/autoint/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo_autoint/slot_test_data_full/* ./test_tipc/data/infer
    fi

elif [ ${model_name} == "ple" ]; then
    # prepare pretrained weights and dataset 
    wget -nc -P  ./test_tipc/save_ple_model https://paddlerec.bj.bcebos.com/tipc/ple.tar
    cd test_tipc/save_ple_model && tar -xvf ple.tar && rm -rf ple.tar && cd ../../

    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/multitask/ple/data/train/* ./test_tipc/data/train
        cp -r ./models/multitask/ple/data/test/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/census
        bash run.sh
        cd ../..
        cp -r ./datasets/census/train_all/* ./test_tipc/data/train
        cp -r ./datasets/census/test_all/* ./test_tipc/data/infer
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/census
        bash run.sh
        cd ../..
        cp -r ./models/multitask/ple/data/train/* ./test_tipc/data/train
        cp -r ./datasets/census/test_all/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/census
        bash run.sh
        cd ../..
        cp -r ./models/multitask/ple/data/train/* ./test_tipc/data/train
        cp -r ./datasets/census/test_all/* ./test_tipc/data/infer
    fi

elif [ ${model_name} == "esmm" ]; then
    # prepare pretrained weights and dataset 
    wget -nc -P  ./test_tipc/save_esmm_model https://paddlerec.bj.bcebos.com/esmm/esmm.tar
    cd test_tipc/save_esmm_model && tar -xvf esmm.tar && rm -rf esmm.tar && cd ../../

    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/multitask/esmm/data/train/* ./test_tipc/data/train
        cp -r ./models/multitask/esmm/data/test/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/ali-ccp
        bash run.sh
        cd ../..
        cp -r ./datasets/ali-ccp/train_data/* ./test_tipc/data/train
        cp -r ./datasets/ali-ccp/test_data/* ./test_tipc/data/infer
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/ali-ccp
        bash run.sh
        cd ../..
        cp -r ./models/multitask/esmm/data/train/* ./test_tipc/data/train
        cp -r ./datasets/ali-ccp/test_data/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/ali-ccp
        bash run.sh
        cd ../..
        cp -r ./models/multitask/esmm/data/train/* ./test_tipc/data/train
        cp -r ./datasets/ali-ccp/test_data/* ./test_tipc/data/infer
    fi


elif [ ${model_name} == "dssm" ]; then
    # prepare pretrained weights and dataset 
    # 占位
    # 占位

    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/match/dssm/data/train/* ./test_tipc/data/train
        cp -r ./models/match/dssm/data/test/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/BQ_dssm
        bash run.sh
        cd ../..
        cp -r ./datasets/BQ_dssm/big_train/* ./test_tipc/data/train
        cp -r ./datasets/BQ_dssm/big_test/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/BQ_dssm
        bash run.sh
        cd ../..
        cp -r ./models/match/dssm/data/train/* ./test_tipc/data/train
        cp -r ./datasets/BQ_dssm/big_test/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/BQ_dssm
        bash run.sh
        cd ../..
        cp -r ./models/match/dssm/data/train/* ./test_tipc/data/train
        cp -r ./datasets/BQ_dssm/big_test/* ./test_tipc/data/infer
    fi

elif [ ${model_name} == "mmoe" ]; then
    # prepare pretrained weights and dataset 
    wget -nc -P  ./test_tipc/save_mmoe_model https://paddlerec.bj.bcebos.com/mmoe/mmoe.tar
    cd test_tipc/save_mmoe_model && tar -xvf mmoe.tar && rm -rf mmoe.tar && cd ../../

    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/multitask/mmoe/data/train/* ./test_tipc/data/train
        cp -r ./models/multitask/mmoe/data/test/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/census
        bash run.sh
        cd ../..
        cp -r ./datasets/census/train_all/* ./test_tipc/data/train
        cp -r ./datasets/census/test_all/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/census
        bash run.sh
        cd ../..
        cp -r ./models/multitask/mmoe/data/train/* ./test_tipc/data/train
        cp -r ./datasets/census/test_all/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/census
        bash run.sh
        cd ../..
        cp -r ./models/multitask/mmoe/data/train/* ./test_tipc/data/train
        cp -r ./datasets/census/test_all/* ./test_tipc/data/infer
    fi

elif [ ${model_name} == "dlrm" ]; then
    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/rank/dlrm/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./models/rank/dlrm/data/sample_data/train/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./datasets/criteo/slot_train_data_full/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./models/rank/dlrm/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/criteo
        bash run.sh
        cd ../..
        cp -r ./models/rank/dlrm/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./datasets/criteo/slot_test_data_full/* ./test_tipc/data/infer
    fi

elif [ ${model_name} == "ensfm" ]; then
    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/recall/ensfm/data/sample_data/* ./test_tipc/data/train
        cp -r ./models/recall/ensfm/data/sample_data/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/ml-1m_ensfm
        bash run.sh
        cd ../..
        cp -r ./datasets/ml-1m_ensfm/data/ml-1m-ensfm/train.csv ./test_tipc/data/train
        cp -r ./datasets/ml-1m_ensfm/data/ml-1m-ensfm/test.csv ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/ml-1m_ensfm
        bash run.sh
        cd ../..
        cp -r ./models/recall/ensfm/data/sample_data/* ./test_tipc/data/train
        cp -r ./datasets/ml-1m_ensfm/data/ml-1m-ensfm/test.csv ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/ml-1m_ensfm
        bash run.sh
        cd ../..
        cp -r ./models/recall/ensfm/data/sample_data/* ./test_tipc/data/train
        cp -r ./datasets/ml-1m_ensfm/data/ml-1m-ensfm/test.csv ./test_tipc/data/infer
    fi

elif [ ${model_name} == "tisas" ]; then
    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/recall/tisas/data/sample_data/* ./test_tipc/data/train
        cp -r ./models/recall/tisas/data/sample_data/* ./test_tipc/data/infer
        echo "demo data ready"
    fi

elif [ ${model_name} == "dselect_k" ]; then
    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/multitask/dselect_k/data/* ./test_tipc/data/train
        cp -r ./models/multitask/dselect_k/data/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/Multi_MNIST_DselectK
        bash run.sh
        cd ../..
        cp -r ./datasets/Multi_MNIST_DselectK/train/* ./test_tipc/data/train
        cp -r ./datasets/Multi_MNIST_DselectK/test/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/Multi_MNIST_DselectK
        bash run.sh
        cd ../..
        cp -r ./models/multitask/dselect_k/data/* ./test_tipc/data/train
        cp -r ./datasets/Multi_MNIST_DselectK/test/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/Multi_MNIST_DselectK
        bash run.sh
        cd ../..
        cp -r ./models/multitask/dselect_k/data/* ./test_tipc/data/train
        cp -r ./datasets/Multi_MNIST_DselectK/test/* ./test_tipc/data/infer
    fi

elif [ ${model_name} == "dsin" ]; then
    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/rank/dsin/data/sample_data/* ./test_tipc/data/train
        cp -r ./models/rank/dsin/data/sample_data/* ./test_tipc/data/infer
        echo "demo data ready"
    fi

elif [ ${model_name} == "aitm" ]; then
    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/multitask/aitm/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./models/multitask/aitm/data/sample_data/test/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/ali-cpp_aitm
        bash run.sh
        cd ../..
        cp -r ./datasets/ali-cpp_aitm/whole_data/train/* ./test_tipc/data/train
        cp -r ./datasets/ali-cpp_aitm/whole_data/test/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/ali-cpp_aitm
        bash run.sh
        cd ../..
        cp -r ./datasets/ali-cpp_aitm/whole_data/train/* ./test_tipc/data/train
        cp -r ./datasets/ali-cpp_aitm/whole_data/test/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/ali-cpp_aitm
        bash run.sh
        cd ../..
        cp -r ./datasets/ali-cpp_aitm/whole_data/train/* ./test_tipc/data/train
        cp -r ./datasets/ali-cpp_aitm/whole_data/test/* ./test_tipc/data/infer
    fi
elif [ ${model_name} == "sign" ]; then
    rm -rf ./test_tipc/data/*
    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/rank/sign/data/* ./test_tipc/data/train
        cp -r ./models/rank/sign/data/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/sign
        bash run.sh
        cd ../..
        cp -r ./datasets/sign/train/* ./test_tipc/data/train
        cp -r ./datasets/sign/test/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/sign
        bash run.sh
        cd ../..
        cp -r ./models/rank/sign/data/* ./test_tipc/data/train
        cp -r ./datasets/sign/test/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/sign
        bash run.sh
        cd ../..
        cp -r ./models/rank/sign/data/* ./test_tipc/data/train
        cp -r ./datasets/sign/test/* ./test_tipc/data/infer
    fi
elif [ ${model_name} == "iprec" ]; then
    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/rank/iprec/data/sample_data/train/* ./test_tipc/data/train
        cp -r ./models/rank/iprec/data/sample_data/test/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/iprec
        bash run.sh
        cd ../..
        cp -r ./datasets/iprec/whole_data/train/* ./test_tipc/data/train
        cp -r ./datasets/iprec/whole_data/test/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/iprec
        bash run.sh
        cd ../..
        cp -r ./datasets/iprec/whole_data/train/* ./test_tipc/data/train
        cp -r ./datasets/iprec/whole_data/test/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/iprec
        bash run.sh
        cd ../..
        cp -r ./datasets/iprec/whole_data/train/* ./test_tipc/data/train
        cp -r ./datasets/iprec/whole_data/test/* ./test_tipc/data/infer
    fi
elif [ ${model_name} == "kim" ]; then
    rm -rf ./test_tipc/data/*
    mkdir -p ./test_tipc/data/train
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/match/kim/data/sample_data/* ./test_tipc/data/train
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/kim
        bash run.sh
        cd ../..
        cp -r ./datasets/kim/data/whole_data/* ./test_tipc/data/train
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/kim
        bash run.sh
        cd ../..
        cp -r ./datasets/kim/data/whole_data/* ./test_tipc/data/train
        echo "whole data ready"
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/kim
        bash run.sh
        cd ../..
        cp -r ./datasets/kim/data/whole_data/* ./test_tipc/data/train
        echo "whole data ready"
    fi
elif [ ${model_name} == "fgcnn" ]; then
    rm -rf ./test_tipc/data/*
    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/rank/fgcnn/data/trainlite/* ./test_tipc/data/train
        cp -r ./models/rank/fgcnn/data/testlite/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/criteo_fgcnn
        bash run.sh
        cd ../..
        cp -r ./datasets/criteo_fgcnn/train/train.h5 ./test_tipc/data/train
        cp -r ./datasets/criteo_fgcnn/test/valid.h5 ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/criteo_fgcnn
        bash run.sh
        cd ../..
        cp -r ./datasets/criteo_fgcnn/train/train.h5 ./test_tipc/data/train
        cp -r ./datasets/criteo_fgcnn/test/valid.h5 ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/criteo_fgcnn
        bash run.sh
        cd ../..
        cp -r ./models/rank/fgcnn/data/trainlite/* ./test_tipc/data/train
        cp -r ./datasets/criteo_fgcnn/test/valid.h5 ./test_tipc/data/infer
    fi
elif [ ${model_name} == "dpin" ]; then
    rm -rf ./test_tipc/data/*
    mkdir -p ./test_tipc/data/train
    mkdir -p ./test_tipc/data/infer
    if [ ${MODE} = "lite_train_lite_infer" ];then
        cp -r ./models/rank/dpin/data/* ./test_tipc/data/train
        cp -r ./models/rank/dpin/data/* ./test_tipc/data/infer
        echo "demo data ready"
    elif [ ${MODE} = "whole_train_whole_infer" ];then
        cd ./datasets/KDD2012_track2
        bash run.sh
        cd ../..
        cp -r ./datasets/KDD2012_track2/train/* ./test_tipc/data/train
        cp -r ./datasets/KDD2012_track2/test/* ./test_tipc/data/infer
        echo "whole data ready"
    elif [ ${MODE} = "whole_infer" ];then
        cd ./datasets/KDD2012_track2
        bash run.sh
        cd ../..
        cp -r ./datasets/KDD2012_track2/train/* ./test_tipc/data/train
        cp -r ./datasets/KDD2012_track2/test/* ./test_tipc/data/infer
    elif [ ${MODE} = "lite_train_whole_infer" ];then
        cd ./datasets/KDD2012_track2
        bash run.sh
        cd ../..
        cp -r ./models/rank/dpin/data/* ./test_tipc/data/train
        cp -r ./datasets/KDD2012_track2/test/* ./test_tipc/data/infer
    fi
elif [ ${model_name} == "deep_walk" ]; then
    if [ ${MODE} = "benchmark_train" ];then
        python -m pip install paddlenlp==2.0.0rc16
        python -m pip install protobuf==3.20.0 -U
        python -m pip install pgl -U
        python -m pip install gpustat==1.0.0 -U
    fi
fi
