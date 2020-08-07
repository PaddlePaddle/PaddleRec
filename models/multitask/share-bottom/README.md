# Share_bottom
以下是本例的简要目录结构及说明： 

```
├── data # 文档
	├── train #训练数据
		├── train_data.txt
	├── test  #测试数据
		├── test_data.txt
	├── run.sh
	├── data_preparation.py
├── __init__.py 
├── config.yaml #配置文件
├── census_reader.py #数据读取文件
├── model.py #模型文件

```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

## 内容

- [模型简介](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/share-bottom#模型简介)
- [数据准备](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/share-bottom#数据准备)
- [运行环境](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/share-bottom#运行环境)
- [快速开始](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/share-bottom#快速开始)
- [论文复现](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/share-bottom#论文复现)
- [进阶使用](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/share-bottom#进阶使用)
- [FAQ](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/share-bottom#FAQ)

## 模型简介

share_bottom是多任务学习的基本框架，其特点是对于不同的任务，底层的参数和网络结构是共享的，这种结构的优点是极大地减少网络的参数数量的情况下也能很好地对多任务进行学习，但缺点也很明显，由于底层的参数和网络结构是完全共享的，因此对于相关性不高的两个任务会导致优化冲突，从而影响模型最终的结果。后续很多Neural-based的多任务模型都是基于share_bottom发展而来的，如MMOE等模型可以改进share_bottom在多任务之间相关性低导致模型效果差的缺点。

我们在Paddlepaddle实现share_bottom网络结构，并在开源数据集Census-income Data上验证模型效果。两个任务的auc分别为：

1.income

>max_sb_test_auc_income：0.94993
>
>mean_sb_test_auc_income： 0.93120

2.marital

> max_sb_test_auc_marital：0.99384
>
> mean_sb_test_auc_marital：0.99256

本项目在paddlepaddle上实现share_bottom的网络结构，并在开源数据集 [Census-income Data](https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD) )上验证模型效果, 本模型配置默认使用demo数据集，若进行精度验证，请参考[论文复现](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/share-bottom#论文复现)部分。

本项目支持功能

训练：单机CPU、单机单卡GPU、单机多卡GPU、本地模拟参数服务器训练、增量训练，配置请参考 [启动训练](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/train.md)

预测：单机CPU、单机单卡GPU ；配置请参考[PaddleRec 离线预测](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/predict.md)

## 数据准备

数据地址： [Census-income Data](https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD) )

数据解压后， 在create_data.sh脚本文件中添加文件的路径，并运行脚本。

```sh
mkdir train_data
mkdir test_data
mkdir data
train_path="data/census-income.data"
test_path="data/census-income.test"
train_data_path="train_data/"
test_data_path="test_data/"
pip install -r requirements.txt
wget -P data/ https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz
tar -zxvf data/census.tar.gz -C data/

python data_preparation.py --train_path ${train_path} \
                           --test_path ${test_path} \
                           --train_data_path ${train_data_path}\
                           --test_data_path ${test_data_path}

```

生成的格式以逗号为分割点

```
0,0,73,0,0,0,0,1700.09,0,0
```



## 运行环境

PaddlePaddle>=1.7.2

python 2.7/3.5/3.6/3.7

PaddleRec >=0.1

os : windows/linux/macos

## 快速开始

### 单机训练

CPU环境

在config.yaml文件中设置好设备，epochs等。

```sh
dataset:
- name: dataset_train
  batch_size: 5
  type: QueueDataset
  data_path: "{workspace}/data/train"
  data_converter: "{workspace}/census_reader.py"
- name: dataset_infer
  batch_size: 5
  type: QueueDataset
  data_path: "{workspace}/data/train"
  data_converter: "{workspace}/census_reader.py"
```

### 单机预测

CPU环境

在config.yaml文件中设置好epochs、device等参数。

```sh
- name: infer_runner
  class: infer
  init_model_path: "increment/0"
  device: cpu
```

## 论文复现

用原论文的完整数据复现论文效果需要在config.yaml中修改batch_size=32, thread_num=8, epoch_num=100

使用gpu p100 单卡训练 4.5h 100轮， 测试auc:best: 0.9939,mean:0.9931 

修改后运行方案：修改config.yaml中的'workspace'为config.yaml的目录位置，执行

```text
python -m paddlerec.run -m /home/your/dir/config.yaml #调试模式 直接指定本地config的绝对路径
```

## 进阶使用

## FAQ
