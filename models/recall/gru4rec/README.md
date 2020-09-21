# GRU4REC

以下是本例的简要目录结构及说明： 

```
├── data #样例数据及数据处理相关文件
	├── train
		├── small_train.txt # 样例训练数据
  ├── test
		├── small_test.txt # 样例测试数据
  ├── convert_format.py # 数据转换脚本
  ├── download.py # 数据下载脚本
  ├── preprocess.py # 数据预处理脚本
├── __init__.py
├── README.md # 文档
├── model.py #模型文件
├── config.yaml #配置文件
├── data_prepare.sh #一键数据处理脚本
├── rsc15_reader.py #reader
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
GRU4REC模型的介绍可以参阅论文[Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)。

论文的贡献在于首次将RNN（GRU）运用于session-based推荐，相比传统的KNN和矩阵分解，效果有明显的提升。

论文的核心思想是在一个session中，用户点击一系列item的行为看做一个序列，用来训练RNN模型。预测阶段，给定已知的点击序列作为输入，预测下一个可能点击的item。

session-based推荐应用场景非常广泛，比如用户的商品浏览、新闻点击、地点签到等序列数据。

本模型配置默认使用demo数据集，若进行精度验证，请参考[论文复现](#论文复现)部分。

本项目支持功能

训练：单机CPU、单机单卡GPU、本地模拟参数服务器训练、增量训练，配置请参考 [启动训练](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/train.md)   

预测：单机CPU、单机单卡GPU；配置请参考[PaddleRec 离线预测](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/predict.md) 

## 数据处理
本示例中数据处理共包含三步：
- Step1: 原始数据数据集下载
```
cd data/
python download.py
```
- Step2: 数据预处理及格式转换。
  1. 以session_id为key合并原始数据集，得到每个session的日期，及顺序点击列表。
  2. 过滤掉长度为1的session；过滤掉点击次数小于5的items。
  3. 训练集、测试集划分。原始数据集里最新日期七天内的作为训练集，更早之前的数据作为测试集。
```
python preprocess.py
python convert_data.py
```
这一步之后，会在data/目录下得到两个文件，rsc15_train_tr_paddle.txt为原始训练文件，rsc15_test_paddle.txt为原始测试文件。格式如下所示：
```
214536502 214536500 214536506 214577561
214662742 214662742 214825110 214757390 214757407 214551617
214716935 214774687 214832672
214836765 214706482
214701242 214826623
214826835 214826715
214838855 214838855
214576500 214576500 214576500
214821275 214821275 214821371 214821371 214821371 214717089 214563337 214706462 214717436 214743335 214826837 214819762
214717867 21471786
```
- Step3: 数据整理。将训练文件统一放在data/all_train目录下，测试文件统一放在data/all_test目录下。
```
mkdir raw_train_data && mkdir raw_test_data
mv rsc15_train_tr_paddle.txt raw_train_data/ && mv rsc15_test_paddle.txt raw_test_data/
mkdir all_train && mkdir all_test

python text2paddle.py raw_train_data/ raw_test_data/ all_train all_test vocab.txt
```

方便起见，我们提供了一键式数据生成脚本：
```
sh data_prepare.sh
```

## 运行环境

PaddlePaddle>=1.7.2 

python 2.7/3.5/3.6/3.7

PaddleRec >=0.1

os : windows/linux/macos

## 快速开始

### 单机训练

```
mode: [cpu_train_runner, cpu_infer_runner]

runner:
- name: cpu_train_runner
  class: train
  device: cpu
  epochs: 10
  save_checkpoint_interval: 2
  save_inference_interval: 4
  save_checkpoint_path: "increment_gru4rec"
  save_inference_path: "inference_gru4rec"
  print_interval: 10
  phase: train
- name: cpu_infer_runner
  class: infer
  init_model_path: "increment_gru4rec"
  device: cpu
  phase: infer
```

### 单机预测

### 运行
```
python -m paddlerec.run -m paddlerec.models.recall.w2v
```

### 结果展示

样例数据训练结果展示：

```
Running SingleStartup.
Running SingleRunner.
batch: 1, acc: [0.03125]
batch: 2, acc: [0.0625]
batch: 3, acc: [0.]
...
epoch 0 done, use time: 0.0605320930481, global metrics: acc=[0.]
...
epoch 19 done, use time: 0.33447098732, global metrics: acc=[0.]
```

样例数据预测结果展示:
```
user:0, top K videos:[40, 31, 4, 33, 93]
user:1, top K videos:[35, 57, 58, 40, 17]
user:2, top K videos:[35, 17, 88, 40, 9]
user:3, top K videos:[73, 35, 39, 58, 38]
user:4, top K videos:[40, 31, 57, 4, 73]
user:5, top K videos:[38, 9, 7, 88, 22]
user:6, top K videos:[35, 73, 14, 58, 28]
user:7, top K videos:[35, 73, 58, 38, 56]
user:8, top K videos:[38, 40, 9, 35, 99]
user:9, top K videos:[88, 73, 9, 35, 28]
user:10, top K videos:[35, 52, 28, 54, 73]
```

## 进阶使用

## FAQ
