# GRU4REC

代码请参考：[GRU4REC模型](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/recall/gru4rec/)

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
python convert_format.py
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
- Step3: 生成字典并整理数据路径。这一步会根据训练和测试文件生成字典和对应的paddle输入文件，并将训练文件统一放在data/all_train目录下，测试文件统一放在data/all_test目录下。
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

在config.yaml文件中设置好设备，epochs等。
```
runner:
- name: cpu_train_runner
  class: train
  device: cpu  # gpu
  epochs: 10
  save_checkpoint_interval: 1
  save_inference_interval: 1
  save_checkpoint_path: "increment_gru4rec"
  save_inference_path: "inference_gru4rec"
  save_inference_feed_varnames: ["src_wordseq", "dst_wordseq"] # feed vars of save inference
  save_inference_fetch_varnames: ["mean_0.tmp_0", "top_k_0.tmp_0"]
  print_interval: 10
  phases: [train]

```

### 单机预测

在config.yaml文件中设置好设备，epochs等。
```
- name: cpu_infer_runner
  class: infer
  init_model_path: "increment_gru4rec"
  device: cpu  # gpu
  phases: [infer]
```

### 运行
```
python -m paddlerec.run -m models/recall/gru4rec/config.yaml
```

### 结果展示

样例数据训练结果展示：

```
Running SingleStartup.
Running SingleRunner.
2020-09-22 03:31:18,167-INFO:   [Train],  epoch: 0,  batch: 10, time_each_interval: 4.34s, RecallCnt: [1669.], cost: [8.366313], InsCnt: [16228.], Acc(Recall@20): [0.10284693]
2020-09-22 03:31:21,982-INFO:   [Train],  epoch: 0,  batch: 20, time_each_interval: 3.82s, RecallCnt: [3168.], cost: [8.170701], InsCnt: [31943.], Acc(Recall@20): [0.09917666]
2020-09-22 03:31:25,797-INFO:   [Train],  epoch: 0,  batch: 30, time_each_interval: 3.81s, RecallCnt: [4855.], cost: [8.017181], InsCnt: [47892.], Acc(Recall@20): [0.10137393]
...
epoch 0 done, use time: 6003.78719687, global metrics: cost=[4.4394927], InsCnt=23622448.0 RecallCnt=14547467.0 Acc(Recall@20)=0.6158323218660487
2020-09-22 05:11:17,761-INFO:   save epoch_id:0 model into: "inference_gru4rec/0"
...
epoch 9 done, use time: 6009.97707605, global metrics: cost=[4.069373], InsCnt=236237470.0 RecallCnt=162838200.0 Acc(Recall@20)=0.6892988086157644
2020-09-22 20:17:11,358-INFO:   save epoch_id:9 model into: "inference_gru4rec/9"
PaddleRec Finish
```

样例数据预测结果展示:
```
Running SingleInferStartup.
Running SingleInferRunner.
load persistables from increment_gru4rec/9
2020-09-23 03:46:21,081-INFO:   [Infer] batch: 20, time_each_interval: 3.68s, RecallCnt: [24875.], InsCnt: [35581.], Acc(Recall@20): [0.6991091]
Infer infer of epoch 9 done, use time: 5.25408315659, global metrics: InsCnt=52551.0 RecallCnt=36720.0 Acc(Recall@20)=0.698749785922247
...
Infer infer of epoch 0 done, use time: 5.20699501038, global metrics: InsCnt=52551.0 RecallCnt=33664.0 Acc(Recall@20)=0.6405967536298073
PaddleRec Finish
```

## 论文复现

用原论文的完整数据复现论文效果需要在config.yaml修改超参：
- batch_size: 修改config.yaml中dataset_train数据集的batch_size为500。
- epochs: 修改config.yaml中runner的epochs为10。
- 数据源：修改config.yaml中dataset_train数据集的data_path为"{workspace}/data/all_train"，dataset_test数据集的data_path为"{workspace}/data/all_test"。

使用gpu训练10轮 测试结果为

epoch | 测试recall@20 | 速度(s)
-- | -- | --
1 | 0.6406 | 6003
2 | 0.6727 | 6007
3 | 0.6831 | 6108
4 | 0.6885 | 6025
5 | 0.6913 | 6019
6 | 0.6931 | 6011
7 | 0.6952 | 6015
8 | 0.6968 | 6076
9 | 0.6972 | 6076
10 | 0.6987| 6009

修改后运行方案：修改config.yaml中的'workspace'为config.yaml的目录位置，执行
```
python -m paddlerec.run -m /home/your/dir/config.yaml #调试模式 直接指定本地config的绝对路径
```

## 进阶使用

## FAQ
