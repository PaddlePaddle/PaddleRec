#!/bin/bash
source test_tipc/common_func.sh

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

function func_parser_config() {
    strs=$1
    IFS=" "
    array=(${strs})
    tmp=${array[3]}
    echo ${tmp}
}

function func_parser_execute_python() {
    strs=$1
    IFS=" "
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}

BASEDIR=$(dirname "$0")
REPO_ROOT_PATH=$(readlinkf ${BASEDIR}/../)

FILENAME=$1

# change gpu to npu in tipc txt configs
sed -i "s/runner.use_gpu/runner.use_npu/g" $FILENAME
sed -i "s/--use_gpu/--use_npu/g" $FILENAME
sed -i "s/--enable_tensorRT:False|True/--enable_tensorRT:False/g" $FILENAME
sed -i "s/--enable_tensorRT:True|False/--enable_tensorRT:False/g" $FILENAME
sed -i "s/--benchmark:True/--benchmark:False/g" $FILENAME
dataline=`cat $FILENAME`

# change gpu to npu in execution script
sed -i 's/\"gpu\"/\"npu\"/g' test_tipc/test_train_inference_python.sh

# parser params
IFS=$'\n'
lines=(${dataline})

# replace gpu to npu in trainer.py
trainer_py=$(func_parser_value "${lines[15]}")
trainer_config=$(func_parser_execute_python ${trainer_py})
sed -i 's/config.get(\"runner.use_gpu\", True)/config.get(\"runner.use_gpu\", False)/g' "$REPO_ROOT_PATH/$trainer_config"

# replace gpu to npu in to_static.py
to_static_py=$(func_parser_value "${lines[29]}")
to_static_config=$(func_parser_execute_python ${to_static_py})
sed -i 's/use_gpu/use_npu/g' "$REPO_ROOT_PATH/$to_static_config"
sed -i 's/'"'"'gpu'"'"'/'"'"'npu'"'"'/g' "$REPO_ROOT_PATH/$to_static_config"

# replace gpu to npu in paddle_infer.py
inference_py=$(func_parser_value "${lines[39]}")
inference_config=$(func_parser_execute_python ${inference_py})
if [[ $inference_config =~ "test_tipc" ]]; then
    sed -i 's/config.enable_use_gpu(1000, 0)/config.enable_npu()/g' "$REPO_ROOT_PATH/$inference_config"
    sed -i 's/use_gpu/use_npu/g' "$REPO_ROOT_PATH/$inference_config"
    sed -i 's/'"'"'gpu'"'"'/'"'"'npu'"'"'/g' "$REPO_ROOT_PATH/$inference_config"
else
    sed -i 's/"--use_gpu", type=str2bool, default=True/"--use_gpu", type=str2bool, default=False/g' "$REPO_ROOT_PATH/$inference_config"
fi

# replace training config file
grep -n './models/.*yaml' $FILENAME  | cut -d ":" -f 1 \
| while read line_num ; do 
    train_cmd=$(func_parser_value "${lines[line_num-1]}")
    trainer_config=$(func_parser_config ${train_cmd})
    echo $trainer_config
    sed -i 's/use_gpu/use_npu/g' "$REPO_ROOT_PATH/$trainer_config"
done

# pass parameters to test_train_inference_python.sh
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $2"
echo $cmd
eval $cmd
