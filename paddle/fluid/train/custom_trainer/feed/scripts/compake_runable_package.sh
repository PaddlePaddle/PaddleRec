#!/bin/bash
#用于运行期的hadoop访问
TRAINER_HODOOP_HOME=""
#用于跟据网络脚本生成模型
TRAINER_PYTHON_HOME="/home/xiexionghang/paddle/py-paddle/"

#环境准备
if [ ! -f ${TRAINER_PYTHON_HOME}/python/bin/paddle ];then
    echo "Miss File: ${TRAINER_PYTHON_HOME}/python/bin/paddle"
    echo "TRAINER_PYTHON_HOME:${TRAINER_PYTHON_HOME} is invalid, Fix it, or Get From here:"
    echo "wget ftp://cp01-arch-gr06.epc.baidu.com/home/xiexionghang/paddle/py-paddle.tar.gz" 
    echo "Then set TRAINER_PYTHON_HOME"
    exit 0
fi
TRAINER_PYTHON_BIN=${TRAINER_PYTHON_HOME}/python/bin/python
# for bad paddle  这里需要想办法解决，paddle的前置目录太多
if [ ! -f ../../../third_party/install/pslib/lib/libps.so ];then
    mkdir -p ../../../third_party/install/pslib/lib/
    ln -s ${TRAINER_PYTHON_HOME}/third_party/install/pslib/lib/libps.so ../../../third_party/install/pslib/lib/libps.so
fi


#生成模型配置
#这里按名匹配 可能会出现匹配错误&兼容性差的问题，最好是先python解析yaml文件
items=`grep " name:" conf/trainer.yaml | awk -F ':' '{print $2}' |awk '{sub("^ *","");sub(" *$","");print}'`
for item in ${items[@]};
do
    if [ ! -f scripts/${item}.py ];then
        echo "Missing model_net config: scripts/${item}.py, skip it $item" 
        continue
    fi
    rm -rf model/$item
    ${TRAINER_PYTHON_BIN} scripts/create_programs.py scripts/${item}.py
    if [ $? -ne 0 ];then
        echo "Create model with scripts/${item}.py failed" 
        exit 1
    fi
done

#输出package包
rm -rf package
mkdir package
cp -r bin conf tool scripts model so package
cp -r ${TRAINER_HODOOP_HOME} package/hadoop-client
