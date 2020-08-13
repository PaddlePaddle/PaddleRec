# look-alike recall

 以下是本例的简要目录结构及说明： 

```
├── config.yaml # 配置文件
├── data # 样例数据文件夹
│   ├── build_dataset.py # 生成样例数据程序示例
│   └── train_data # 样例数据 
│       └── paddle_train.txt # 默认样例数据
├── __init__.py
├── model.py # 模型文件
└── README.md # 文档
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)


---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [论文复现](#论文复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)

## 模型简介

本目录目录模型文件参考论文 [《Real-time Attention Based Look-alike Model for Recommender System》]( https://arxiv.org/pdf/1906.05022)，是发表在KDD 19上的一篇论文，文章指出目前基于深度学习的模型没有能够缓解"马太效应"，模型倾向于偏向拥有比较多的特征的头部资源，而那些同样优质的缺少用户交互信息的长尾资源得不到充分的曝光。文章提出推荐广告的经典算法 look-alike 是缓解"马太效应"一个不错的选择。但是受限于推荐领域有别于推荐广告严格的实时性要求，该算法未被大规模采用。基于以上，文章提出了一种实时性的基于attention的looka-like算法 RALM。

本项目在paddlepaddle上主要实现RALM的网络结构，其他更多实时性的工程尝试请参考原论文。因为原论文没有在开源数据集上验证模型效果，本项目提供了100行样例数据。验证模型的正确性，若进行精度验证，请参考样例数据格式或者自定义更改相关配置构建自己的数据集，在工程环境中进行验证。

模型大体结构为双塔结构，可以理解为target user 和 user seeds两个塔。使用论文中提出的local_attention 和 global_attention模块。损失函数采用cosine similarity损失函数。

本项目支持功能

训练：单机CPU、单机单卡GPU、单机多卡GPU、本地模拟参数服务器训练、增量训练，配置请参考 [启动训练](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/train.md)   

预测：单机CPU、单机单卡GPU ；配置请参考[PaddleRec 离线预测](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/predict.md) 

## 数据准备

数据地址：[样例数据](./data/train_data/paddle_train.txt)

样例数据中每条样本包含三个slot：user_seeds, target_user, label。
 (1) user_seeds: 基于当前的资源圈定的种子用户
 (2) target_user: 目标用户
 (3) label: 点击与否

注：本项目提供的样例数据为完全fake的，没有任何实际参考价值。用户可根据样例数据格式自行构建基于自己项目或者工程的数据集。

执行build_dataset.py生成训练集和测试集

```
cd data
python build_dataset.py
```

运行后生成的数据格式为3个离散化特征，用'\t'切分, 对应的slot是user_seeds, target_user, label
```
user_seeds:2 user_seeds:3 user_seeds:4 user_seeds:5 user_seeds:6 user_seeds:7 user_seeds:8 user_seeds:9 user_seeds:10 user_seeds:11 user_seeds:12 user_seeds:13 user_seeds:14 user_seeds:15 user_seeds:16 user_seeds:17 user_seeds:18 user_seeds:19 target_user:1 label:1 
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
# select runner by name
mode: [single_cpu_train, single_cpu_infer]
# config of each runner.
# runner is a kind of paddle training class, which wraps the train/infer process.
runner:
- name: single_cpu_train
  class: train
  # num of epochs
  epochs: 4
  # device to run training or infer
  device: cpu
  save_checkpoint_interval: 2 # save model interval of epochs
  save_inference_interval: 4 # save inference
  save_checkpoint_path: "increment_model" # save checkpoint path
  save_inference_path: "inference" # save inference path
  save_inference_feed_varnames: [] # feed vars of save inference
  save_inference_fetch_varnames: [] # fetch vars of save inference
  init_model_path: "" # load model path
  print_interval: 10
  phases: [phase1]
```

### 单机预测

CPU环境

在config.yaml文件中设置好epochs、device等参数。

```
- name: single_cpu_infer
  class: infer
  # num of epochs
  epochs: 1
  # device to run training or infer
  device: cpu #选择预测的设备
  init_model_path: "increment_dnn" # load model path
  phases: [phase2]
```

### 运行
```
python -m paddlerec.run -m models/recall/look-alike_recall/config.yaml
```


### 结果展示

样例数据训练结果展示：

```
PaddleRec: Runner train_runner Begin
Executor Mode: train
processor_register begin
Running SingleInstance.
Running SingleNetwork.
Running SingleStartup.
Running SingleRunner.
I0729 15:51:44.029929 22883 parallel_executor.cc:440] The Program will be executed on CPU using ParallelExecutor, 1 cards are used, so 1 programs are executed in parallel.
I0729 15:51:44.031812 22883 build_strategy.cc:365] SeqOnlyAllReduceOps:0, num_trainers:1
I0729 15:51:44.033733 22883 parallel_executor.cc:307] Inplace strategy is enabled, when build_strategy.enable_inplace = True
I0729 15:51:44.035027 22883 parallel_executor.cc:375] Garbage collection strategy is enabled, when FLAGS_eager_delete_tensor_gb = 0
batch: 1, BATCH_AUC: [0.], AUC: [0.]
batch: 2, BATCH_AUC: [0.], AUC: [0.]
epoch 0 done, use time: 0.0433671474457
PaddleRec Finish
```

## 论文复现

论文中没有提供基于公开数据集的实验结果。

## 进阶使用

## FAQ
