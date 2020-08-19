# GNN

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
	├── train
		├── train.txt
    ├── test
		├── test.txt
	├── download.py
	├── convert_data.py
	├── preprocess.py
├── __init__.py
├── README.md # 文档
├── model.py #模型文件
├── config.yaml #配置文件
├── data_prepare.sh #一键数据处理脚本
├── reader.py #训练数据reader
├── evaluate_reader.py # 预测数据reader
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
SR-GNN模型的介绍可以参阅论文[Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855)。

本文解决的是Session-based Recommendation这一问题,过程大致分为以下四步：

1. 首先对所有的session序列通过有向图进行建模。

2. 然后通过GNN，学习每个node（item）的隐向量表示

3. 通过一个attention架构模型得到每个session的embedding

4. 最后通过一个softmax层进行全表预测

本示例中，我们复现了论文效果，在DIGINETICA数据集上P@20可以达到50.7。

同时推荐用户参考[ IPython Notebook demo](https://aistudio.baidu.com/aistudio/projectDetail/124382)

本模型配置默认使用demo数据集，若进行精度验证，请参考[论文复现](#论文复现)部分。

本项目支持功能

训练：单机CPU、单机单卡GPU、单机多卡GPU、本地模拟参数服务器训练、增量训练，配置请参考 [启动训练](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/train.md)   

预测：单机CPU、单机单卡GPU ；配置请参考[PaddleRec 离线预测](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/predict.md) 

## 数据处理
本示例中数据处理共包含三步：
- Step1: 原始数据数据集下载，本示例提供了两个开源数据集：DIGINETICA和Yoochoose，可选其中任意一个训练本模型。数据下载命令及原始数据格式如下所示。若采用diginetica数据集，执行完该命令之后，会在data目录下得到原始数据文件train-item-views.csv。若采用yoochoose数据集，执行完该命令之后，会在data目录下得到原始数据文件yoochoose-clicks.dat。
    ```
    cd data && python download.py diginetica     # or yoochoose
    ```
    > [Yoochooses](https://2015.recsyschallenge.com/challenge.html)数据集来源于RecSys Challenge 2015，原始数据包含如下字段：
    1. Session ID – the id of the session. In one session there are one or many clicks.
    2. Timestamp – the time when the click occurred.
    3. Item ID – the unique identifier of the item.
    4. Category – the category of the item.

    > [DIGINETICA](https://competitions.codalab.org/competitions/11161#learn_the_details-data2)数据集来源于CIKM Cup 2016 _Personalized E-Commerce Search Challenge_项目。原始数据包含如下字段：
    1. sessionId - the id of the session. In one session there are one or many clicks.
    2. userId - the id of the user, with anonymized user ids.
    3. itemId - the unique identifier of the item.
    4. timeframe - time since the first query in a session, in milliseconds.
    5. eventdate - calendar date.

- Step2: 数据预处理。
    1. 以session_id为key合并原始数据集，得到每个session的日期，及顺序点击列表。
    2. 过滤掉长度为1的session；过滤掉点击次数小于5的items。
    3. 训练集、测试集划分。原始数据集里最新日期七天内的作为训练集，更早之前的数据作为测试集。
    ```
    cd data && python preprocess.py --dataset diginetica   # or yoochoose
    ```
- Step3: 数据整理。 将训练文件统一放在data/train目录下，测试文件统一放在data/test目录下。
    ```
    cat data/diginetica/train.txt | wc -l >> data/config.txt    # or yoochoose1_4 or yoochoose1_64
    rm -rf data/train/*
    rm -rf data/test/*
    mv data/diginetica/train.txt data/train
    mv data/diginetica/test.txt data/test
    ```
数据处理完成后，data/train目录存放训练数据，data/test目录下存放测试数据，数据格式如下:
```
#session\tlabel
10,11,12,12,13,14\t15
```
data/config.txt中存放数据统计信息，第一行代表训练集中item总数，用以配置模型词表大小，第二行代表训练集大小。

方便起见， 我们提供了一键式数据处理脚本：
```
sh data_prepare.sh diginetica      # or yoochoose1_4 or yoochoose1_64
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
  epochs: 2
  # device to run training or infer
  device: cpu
  save_checkpoint_interval: 1 # save model interval of epochs
  save_inference_interval: 1 # save inference
  save_checkpoint_path: "increment_gnn" # save checkpoint path
  save_inference_path: "inference_gnn" # save inference path
  save_inference_feed_varnames: [] # feed vars of save inference
  save_inference_fetch_varnames: [] # fetch vars of save inference
  init_model_path: "" # load model path
  print_interval: 1
  phases: [phase1]
```
### 单机预测

CPU环境

在config.yaml文件中设置好epochs、device等参数。

```
- name: single_cpu_infer
  class: infer
  # device to run training or infer
  device: cpu
  print_interval: 1
  init_model_path: "increment_gnn" # load model path
  phases: [phase2]
```

### 运行
```
python -m paddlerec.run -m models/recall/gnn/config.yaml
```

### 结果展示

样例数据训练结果展示：

```
Running SingleStartup.
Running SingleRunner.
batch: 1, LOSS: [10.67443], InsCnt: [200.], RecallCnt: [0.], Acc(Recall@20): [0.]
batch: 2, LOSS: [10.672471], InsCnt: [300.], RecallCnt: [0.], Acc(Recall@20): [0.]
batch: 3, LOSS: [10.672463], InsCnt: [400.], RecallCnt: [1.], Acc(Recall@20): [0.0025]
batch: 4, LOSS: [10.670724], InsCnt: [500.], RecallCnt: [2.], Acc(Recall@20): [0.004]
batch: 5, LOSS: [10.66949], InsCnt: [600.], RecallCnt: [2.], Acc(Recall@20): [0.00333333]
batch: 6, LOSS: [10.670102], InsCnt: [700.], RecallCnt: [2.], Acc(Recall@20): [0.00285714]
batch: 7, LOSS: [10.671348], InsCnt: [800.], RecallCnt: [2.], Acc(Recall@20): [0.0025]
...
epoch 0 done, use time: 2926.6897077560425, global metrics: LOSS=[6.0788856], InsCnt=719400.0 RecallCnt=224033.0 Acc(Recall@20)=0.3114164581595774
...
epoch 4 done, use time: 3083.101449728012, global metrics: LOSS=[4.249889], InsCnt=3597000.0 RecallCnt=2070666.0 Acc(Recall@20)=0.5756647206005004
```
样例数据预测结果展示:
```
Running SingleInferStartup.
Running SingleInferRunner.
load persistables from increment_gnn/2
batch: 1, InsCnt: [200.], RecallCnt: [96.], Acc(Recall@20): [0.48], LOSS: [5.7198644]
batch: 2, InsCnt: [300.], RecallCnt: [153.], Acc(Recall@20): [0.51], LOSS: [5.4096317]
batch: 3, InsCnt: [400.], RecallCnt: [210.], Acc(Recall@20): [0.525], LOSS: [5.300991]
batch: 4, InsCnt: [500.], RecallCnt: [258.], Acc(Recall@20): [0.516], LOSS: [5.6269655]
batch: 5, InsCnt: [600.], RecallCnt: [311.], Acc(Recall@20): [0.5183333], LOSS: [5.39276]
batch: 6, InsCnt: [700.], RecallCnt: [352.], Acc(Recall@20): [0.50285715], LOSS: [5.633842]
batch: 7, InsCnt: [800.], RecallCnt: [406.], Acc(Recall@20): [0.5075], LOSS: [5.342844]
batch: 8, InsCnt: [900.], RecallCnt: [465.], Acc(Recall@20): [0.51666665], LOSS: [4.918761]
...
Infer phase2 of epoch 0 done, use time: 549.1640813350677, global metrics: InsCnt=60800.0 RecallCnt=31083.0 Acc(Recall@20)=0.511233552631579, LOSS=[5.8957024]
```

## 论文复现

用原论文的完整数据复现论文效果需要在config.yaml修改超参：
- batch_size: 修改config.yaml中dataset_train数据集的batch_size为100。
- epochs: 修改config.yaml中runner的epochs为5。
- sparse_feature_number: 不同训练数据集(diginetica or yoochoose)配置不一致，diginetica数据集配置为43098，yoochoose数据集配置为37484。具体见数据处理后得到的data/config.txt文件中第一行。
- corpus_size: 不同训练数据集配置不一致，diginetica数据集配置为719470，yoochoose数据集配置为5917745。具体见数据处理后得到的data/config.txt文件中第二行。

使用cpu训练 5轮 测试Recall@20:0.51367

修改后运行方案：修改config.yaml中的'workspace'为config.yaml的目录位置，执行
```
python -m paddlerec.run -m /home/your/dir/config.yaml #调试模式 直接指定本地config的绝对路径
```

## 进阶使用

## FAQ
