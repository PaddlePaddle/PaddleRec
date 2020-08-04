# ESMM

以下是本例的简要目录结构及说明： 

```
├── data # 文档
	├── train #训练数据
		├──small.txt
	├── test  #测试数据
		├── small.txt
	├── run.sh
├── __init__.py 
├── config.yaml #配置文件
├── esmm_reader.py #数据读取文件
├── model.py #模型文件
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

## 内容

- [模型简介](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/esmm#模型简介)
- [数据准备](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/esmm#数据准备)
- [运行环境](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/esmm#运行环境)
- [快速开始](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/esmm#快速开始)
- [论文复现](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/esmm#论文复现)
- [进阶使用](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/esmm#进阶使用)
- [FAQ](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/esmm#FAQ)

## 模型简介

不同于CTR预估问题，CVR预估面临两个关键问题：

1. **Sample Selection Bias (SSB)** 转化是在点击之后才“有可能”发生的动作，传统CVR模型通常以点击数据为训练集，其中点击未转化为负例，点击并转化为正例。但是训练好的模型实际使用时，则是对整个空间的样本进行预估，而非只对点击样本进行预估。即是说，训练数据与实际要预测的数据来自不同分布，这个偏差对模型的泛化能力构成了很大挑战。
2. **Data Sparsity (DS)** 作为CVR训练数据的点击样本远小于CTR预估训练使用的曝光样本。

ESMM是发表在 SIGIR’2018 的论文[《Entire Space Multi-Task Model: An Eﬀective Approach for Estimating Post-Click Conversion Rate》](  https://arxiv.org/abs/1804.07931  )文章基于 Multi-Task Learning 的思路，提出一种新的CVR预估模型——ESMM，有效解决了真实场景中CVR预估面临的数据稀疏以及样本选择偏差这两个关键问题

本项目在paddlepaddle上实现ESMM的网络结构，并在开源数据集[Ali-CCP：Alibaba Click and Conversion Prediction](  https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408  )上验证模型效果, 本模型配置默认使用demo数据集，若进行精度验证，请参考[论文复现](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/esmm#论文复现)部分。

本项目支持功能

训练：单机CPU、单机单卡GPU、单机多卡GPU、本地模拟参数服务器训练、增量训练，配置请参考 [启动训练](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/train.md)

预测：单机CPU、单机单卡GPU ；配置请参考[PaddleRec 离线预测](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/predict.md)

## 数据准备

数据地址：[Ali-CCP：Alibaba Click and Conversion Prediction](  https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408  )

```
cd data 
sh run.sh
```

数据格式参见demo数据：data/train


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
    data_converter: "{workspace}/esmm_reader.py"
  - name: dataset_infer
    batch_size: 5
    type: QueueDataset
    data_path: "{workspace}/data/test"
    data_converter: "{workspace}/esmm_reader.py"
```

### 单机预测

CPU环境

在config.yaml文件中设置好epochs、device等参数。

```
 - name: infer_runner
    class: infer
    init_model_path: "increment/1"
    device: cpu
    print_interval: 1
    phases: [infer]
```


## 论文复现

用原论文的完整数据复现论文效果需要在config.yaml中修改batch_size=1000, thread_num=8, epoch_num=4


修改后运行方案：修改config.yaml中的'workspace'为config.yaml的目录位置，执行

```
python -m paddlerec.run -m /home/your/dir/config.yaml #调试模式 直接指定本地config的绝对路径
```

## 进阶使用

## FAQ
