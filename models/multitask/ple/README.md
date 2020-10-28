# PLE

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

- [模型简介](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/ple#模型简介)
- [数据准备](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/ple#数据准备)
- [运行环境](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/ple#运行环境)
- [快速开始](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/ple#快速开始)
- [论文复现](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/ple#论文复现)
- [进阶使用](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/ple#进阶使用)
- [FAQ](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/ple#FAQ)

## 模型简介

多任务模型通过学习不同任务的联系和差异，可提高每个任务的学习效率和质量。但在多任务场景中经常出现跷跷板现象，即有些任务表现良好，有些任务表现变差。  论文[《Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations》](https://dl.acm.org/doi/abs/10.1145/3383313.3412236 ) ，论文提出了Progressive Layered Extraction (简称PLE)，来解决多任务学习的跷跷板现象。 

我们在Paddlepaddle定义PLE的网络结构，在开源数据集Census-income Data上验证模型效果。

若进行精度验证，请参考[论文复现](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/ple#论文复现)部分。

本项目支持功能

训练：单机CPU、单机单卡GPU、单机多卡GPU、本地模拟参数服务器训练、增量训练，配置请参考 [启动训练](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/train.md)
预测：单机CPU、单机单卡GPU ；配置请参考[PaddleRec 离线预测](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/predict.md)

## 数据准备

数据地址： [Census-income Data](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz )


生成的格式以逗号为分割点

```
0,0,73,0,0,0,0,1700.09,0,0
```

完整的大数据参考论文复现部分。

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

## 进阶使用

## FAQ
