# MMOE

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

- [模型简介](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/mmoe#模型简介)
- [数据准备](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/mmoe#数据准备)
- [运行环境](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/mmoe#运行环境)
- [快速开始](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/mmoe#快速开始)
- [论文复现](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/mmoe#论文复现)
- [进阶使用](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/mmoe#进阶使用)
- [FAQ](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/mmoe#FAQ)

## 模型简介

多任务模型通过学习不同任务的联系和差异，可提高每个任务的学习效率和质量。多任务学习的的框架广泛采用shared-bottom的结构，不同任务间共用底部的隐层。这种结构本质上可以减少过拟合的风险，但是效果上可能受到任务差异和数据分布带来的影响。  论文[《Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts》]( https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture- )中提出了一个Multi-gate Mixture-of-Experts(MMOE)的多任务学习结构。MMOE模型刻画了任务相关性，基于共享表示来学习特定任务的函数，避免了明显增加参数的缺点。 

我们在Paddlepaddle定义MMOE的网络结构，在开源数据集Census-income Data上验证模型效果，两个任务的auc分别为：

1.income

> max_mmoe_test_auc_income：0.94937
>
> mean_mmoe_test_auc_income：0.94465

2.marital

> max_mmoe_test_auc_marital：0.99419
>
> mean_mmoe_test_auc_marital：0.99324

若进行精度验证，请参考[论文复现](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/mmoe#论文复现)部分。

本项目支持功能

训练：单机CPU、单机单卡GPU、单机多卡GPU、本地模拟参数服务器训练、增量训练，配置请参考 [启动训练](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/train.md)
预测：单机CPU、单机单卡GPU ；配置请参考[PaddleRec 离线预测](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/predict.md)

## 数据准备

数据地址： [Census-income Data](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz )

数据解压后， 在run.sh脚本文件中添加文件的路径，并运行脚本。

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

```
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

```
- name: infer_runner
  class: infer
  init_model_path: "increment/0"
  device: cpu
```

## 论文复现

用原论文的完整数据复现论文效果需要在config.yaml中修改batch_size=1000, thread_num=8, epoch_num=4

使用gpu p100 单卡训练 6.5h 测试auc: best:0.9940, mean:0.9932

修改后运行方案：修改config.yaml中的'workspace'为config.yaml的目录位置，执行

```
python -m paddlerec.run -m /home/your/dir/config.yaml #调试模式 直接指定本地config的绝对路径
```

## 进阶使用

## FAQ
