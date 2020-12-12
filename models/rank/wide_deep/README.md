# wide&deep

以下是本例的简要目录结构及说明： 

```
├── data # 数据
    ├── sample_data #示例数据
        ├── train #训练数据
            ├── train_data.txt
    ├── create_data.sh #数据下载脚本
    ├── data_preparation.py #数据处理程序
    ├── get_slot_data.py #数据处理程序
    ├── run.sh #一键数据下载脚本
    ├── args.py ## 脚本参数
├── __init__.py 
├── config.yaml #配置文件
├── model.py #模型文件
├── README.md #文档
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [论文复现](#论文复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)

## 模型简介

[《Wide & Deep Learning for Recommender Systems》]( https://arxiv.org/pdf/1606.07792.pdf)是Google 2016年发布的推荐框架，wide&deep设计了一种融合浅层（wide）模型和深层（deep）模型进行联合训练的框架，综合利用浅层模型的记忆能力和深层模型的泛化能力，实现单模型对推荐系统准确性和扩展性的兼顾。从推荐效果和服务性能两方面进行评价：

1. 效果上，在Google Play 进行线上A/B实验，wide&deep模型相比高度优化的Wide浅层模型，app下载率+3.9%。相比deep模型也有一定提升。
2. 性能上，通过切分一次请求需要处理的app 的Batch size为更小的size，并利用多线程并行请求达到提高处理效率的目的。单次响应耗时从31ms下降到14ms。

若进行精度验证，请参考[论文复现](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/wide_deep#论文复现)部分。

本项目支持功能

训练：单机CPU、单机单卡GPU、单机多卡GPU、本地模拟参数服务器训练、增量训练，配置请参考 [启动训练](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/train.md)

预测：单机CPU、单机单卡GPU ；配置请参考[PaddleRec 离线预测](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/predict.md)

## 数据准备

本例在paddlerec上实现wide&deep并在开源数据集Census-income Data上验证模型效果
数据地址： 

[adult.data](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)

[adult.test](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test)

您可以在进入models/rank/wide_deep/data目录，直接运行一键数据生成脚本run.sh获取数据。
```
sh run.sh
```
在本例中需要调用pandas库，如环境中没有提前安装，可以使用命令 pip install pandas 安装。  

运行的结果示例如下：
```
--2020-09-27 16:57:38--  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
Resolving archive.ics.uci.edu... 128.195.10.252
Connecting to archive.ics.uci.edu|128.195.10.252|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 3974305 (3.8M) [application/x-httpd-php]
Saving to: data/adult.data

100%[===================================================================================================================>] 3,974,305   12.6K/s   in 6m 17s

2020-09-27 17:03:57 (10.3 KB/s) - data/adult.data saved [3974305/3974305]

--2020-09-27 17:03:57--  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
Resolving archive.ics.uci.edu... 128.195.10.252
Connecting to archive.ics.uci.edu|128.195.10.252|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 2003153 (1.9M) [application/x-httpd-php]
Saving to: data/adult.test

100%[==================================================================================================================>] 2,003,153   12.7K/s   in 51s

2020-09-27 17:08:04 (13.5 KB/s) - data/adult.test saved [2003153/2003153]
```

## 运行环境

PaddlePaddle>=1.7.2

python 2.7/3.5/3.6/3.7

PaddleRec >=0.1

os : windows/linux/macos


## 快速开始
本文提供了样例数据可以供您快速体验，在paddlerec目录下执行下面的命令即可快速启动训练： 

```
python -m paddlerec.run -m models/rank/wide_deep/config.yaml
```
使用样例数据快速跑通的结果实例:
```
PaddleRec: Runner train_runner Begin
Executor Mode: train
processor_register begin
Running SingleInstance.
Running SingleNetwork.
Warning:please make sure there are no hidden files in the dataset folder and check these hidden files:[]
Running SingleStartup.
Running SingleRunner.
I0927 17:16:18.305258  3437 parallel_executor.cc:440] The Program will be executed on CPU using ParallelExecutor, 1 cards are used, so 1 programs are executed in parallel.
I0927 17:16:18.310783  3437 build_strategy.cc:365] SeqOnlyAllReduceOps:0, num_trainers:1
I0927 17:16:18.314724  3437 parallel_executor.cc:307] Inplace strategy is enabled, when build_strategy.enable_inplace = True
I0927 17:16:18.317752  3437 parallel_executor.cc:375] Garbage collection strategy is enabled, when FLAGS_eager_delete_tensor_gb = 0
2020-09-27 17:16:18,475-INFO:  [Train] batch: 20, time_each_interval: 0.18s, ACC: [0.6], BATCH_AUC: [0.41666667], AUC: [0.61538462]
2020-09-27 17:16:18,583-INFO:  [Train] batch: 40, time_each_interval: 0.11s, ACC: [0.8], BATCH_AUC: [0.875], AUC: [0.59693471]
2020-09-27 17:16:18,625-INFO:  [Train] batch: 60, time_each_interval: 0.04s, ACC: [0.4], BATCH_AUC: [1.], AUC: [0.59405999]
2020-09-27 17:16:18,666-INFO:  [Train] batch: 80, time_each_interval: 0.04s, ACC: [0.8], BATCH_AUC: [0.5], AUC: [0.56687606]
epoch 0 done, use time: 0.503633022308, global metrics: ACC=[1.], BATCH_AUC=[0.], AUC=[0.56696623]
PaddleRec Finish
```

## 论文复现

为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。
在全量数据下模型的指标如下：  

| 模型 | auc | acc | batch_size | thread_num| epoch_num| Time of each epoch |
| :------| :------ | :------ | :------| :------ | :------| :------ | 
| wide_deep | 0.8987 | 0.775 | 40 | 1 | 80 | 约10s |

1. 确认您当前所在目录为PaddleRec/models/rank/wide_deep
2. 在data目录下运行数据一键处理脚本，命令如下：  
``` 
cd data
sh run.sh
cd ..
```
3. 退回deepfm目录中，打开文件config.yaml,更改其中的参数  
将workspace改为您当前的绝对路径。（可用pwd命令获取绝对路径）  
将train_sample中的batch_size从5改为40  
将train_sample中的data_path改为{workspace}/data/slot_train_data  
将infer_sample中的batch_size从5改为40  
将infer_sample中的data_path改为{workspace}/data/slot_test_data  
将train_runner中的epochs改为80
将infer_runner中的init_model_path改为increment/79
4. 运行命令，模型会进行80个epoch的训练，然后预测最后一个epoch，并获得相应auc和acc指标  
```
python -m paddlerec.run -m ./config.yaml
```
5. 经过全量数据训练后，执行预测的结果示例如下：
```
PaddleRec: Runner infer_runner Begin
Executor Mode: infer
processor_register begin
Running SingleInstance.
Running SingleNetwork.
Warning:please make sure there are no hidden files in the dataset folder and check these hidden files:[]
Running SingleInferStartup.
Running SingleInferRunner.
load persistables from increment/79
2020-09-27 17:37:17,679-INFO:  [Infer] batch: 20, time_each_interval: 0.77s, ACC: [0.8], AUC: [0.89880283]
2020-09-27 17:37:18,452-INFO:  [Infer] batch: 40, time_each_interval: 0.77s, ACC: [0.825], AUC: [0.89879974]
2020-09-27 17:37:19,023-INFO:  [Infer] batch: 60, time_each_interval: 0.57s, ACC: [0.7], AUC: [0.89880376]
2020-09-27 17:37:19,591-INFO:  [Infer] batch: 80, time_each_interval: 0.57s, ACC: [0.925], AUC: [0.89879592]
2020-09-27 17:37:20,195-INFO:  [Infer] batch: 100, time_each_interval: 0.60s, ACC: [0.725], AUC: [0.89879213]
2020-09-27 17:37:20,822-INFO:  [Infer] batch: 120, time_each_interval: 0.63s, ACC: [0.775], AUC: [0.89879757]
2020-09-27 17:37:21,303-INFO:  [Infer] batch: 140, time_each_interval: 0.48s, ACC: [0.775], AUC: [0.89879296]
2020-09-27 17:37:21,798-INFO:  [Infer] batch: 160, time_each_interval: 0.49s, ACC: [0.875], AUC: [0.89879267]
2020-09-27 17:37:22,265-INFO:  [Infer] batch: 180, time_each_interval: 0.47s, ACC: [0.85], AUC: [0.89879272]
2020-09-27 17:37:22,835-INFO:  [Infer] batch: 200, time_each_interval: 0.57s, ACC: [0.725], AUC: [0.89878928]
2020-09-27 17:37:23,364-INFO:  [Infer] batch: 220, time_each_interval: 0.53s, ACC: [0.825], AUC: [0.89878807]
2020-09-27 17:37:23,859-INFO:  [Infer] batch: 240, time_each_interval: 0.49s, ACC: [0.7], AUC: [0.8987825]
2020-09-27 17:37:24,337-INFO:  [Infer] batch: 260, time_each_interval: 0.48s, ACC: [0.775], AUC: [0.89878314]
2020-09-27 17:37:24,877-INFO:  [Infer] batch: 280, time_each_interval: 0.54s, ACC: [0.875], AUC: [0.89877827]
2020-09-27 17:37:25,410-INFO:  [Infer] batch: 300, time_each_interval: 0.53s, ACC: [0.75], AUC: [0.89877518]
2020-09-27 17:37:25,985-INFO:  [Infer] batch: 320, time_each_interval: 0.57s, ACC: [0.75], AUC: [0.89876936]
2020-09-27 17:37:26,447-INFO:  [Infer] batch: 340, time_each_interval: 0.46s, ACC: [0.775], AUC: [0.89876268]
2020-09-27 17:37:26,725-INFO:  [Infer] batch: 360, time_each_interval: 0.28s, ACC: [0.75], AUC: [0.8987574]
2020-09-27 17:37:26,889-INFO:  [Infer] batch: 380, time_each_interval: 0.16s, ACC: [0.8], AUC: [0.89874688]
2020-09-27 17:37:27,065-INFO:  [Infer] batch: 400, time_each_interval: 0.18s, ACC: [0.8], AUC: [0.89875484]
Infer infer_phase of epoch increment/79 done, use time: 10.2139520645, global metrics: ACC=[0.775], AUC=[0.89875217]
PaddleRec Finish
```

## 进阶使用

```
# 进入模型目录
cd models/rank/wide_deep 
# 训练
python -u train.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 预测
python -u infer.py -m config.yaml 
```
## FAQ
