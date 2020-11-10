#!/bin/bash
# download paddlerec
hadoop fs -D fs.default.name=afs://yinglong.afs.baidu.com:9902 -D hadoop.job.ugi=paddle,paddle -get /user/paddle/chengmo/paddlerec/whl/ps_benchmark/PaddleRec.tar.gz
tar -xf PaddleRec.tar.gz

# install paddlerec
cd PaddleRec
python setup.py install

# uninstall paddlepaddle
pip uninstall -y paddlepaddle
pip uninstall -y paddlepaddle-gpu

# downlaod paddlepaddle
hadoop fs -D hadoop.job.ugi=paddle,paddle -D fs.default.name=afs://yinglong.afs.baidu.com:9902 -get /user/paddle/chengmo/whl/heter_ps_1029/paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64.whl ./

# install paddlepaddle
pip install paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64.whl --index-url=http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com
