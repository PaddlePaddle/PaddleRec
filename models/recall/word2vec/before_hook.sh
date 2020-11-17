#!/bin/bash
# download paddlerec

# install paddlerec
cd PaddleRec
python setup.py install

# uninstall paddlepaddle
pip uninstall -y paddlepaddle
pip uninstall -y paddlepaddle-gpu

# downlaod paddlepaddle

# install paddlepaddle
pip install paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64.whl --index-url=http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com
