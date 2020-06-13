# 多任务学习模型库

## 简介
我们提供了常见的多任务学习中使用的模型算法的PaddleRec实现, 单机训练&预测效果指标以及分布式训练&预测性能指标等。实现的多任务模型包括 [MMoE](mmoe)、[Share-Bottom](share-bottom)、[ESMM](esmm)。

模型算法库在持续添加中，欢迎关注。

## 目录
* [整体介绍](#整体介绍)
    * [多任务模型列表](#多任务模型列表)
* [使用教程](#使用教程)
    * [数据处理](#数据处理)
    * [训练](#训练)
    * [预测](#预测)
* [效果对比](#效果对比)
    * [模型效果列表](#模型效果列表)

## 整体介绍
### 多任务模型列表

|       模型        |       简介        |       论文        |
| :------------------: | :--------------------: | :--------- |
| Share-Bottom | share-bottom | [1998][Multitask learning](http://reports-archive.adm.cs.cmu.edu/anon/1997/CMU-CS-97-203.pdf) |
| ESMM | Entire Space Multi-Task Model | [SIGIR 2018][Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931) |
| MMOE | Multi-gate Mixture-of-Experts | [KDD 2018][Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007) |

下面是每个模型的简介（注：图片引用自链接中的论文）


[ESMM](https://arxiv.org/abs/1804.07931):
<p align="center">
<img align="center" src="../../doc/imgs/esmm.png">
<p>

[Share-Bottom](http://reports-archive.adm.cs.cmu.edu/anon/1997/CMU-CS-97-203.pdf):
<p align="center">
<img align="center" src="../../doc/imgs/share-bottom.png">
<p>

[MMoE](https://dl.acm.org/doi/abs/10.1145/3219819.3220007):
<p align="center">
<img align="center" src="../../doc/imgs/mmoe.png">
<p>

## 使用教程(快速开始)
```shell
python -m paddlerec.run -m paddlerec.models.multitask.mmoe # mmoe
python -m paddlerec.run -m paddlerec.models.multitask.share-bottom # share-bottom
python -m paddlerec.run -m paddlerec.models.multitask.esmm # esmm
```

## 使用教程（复现论文）
### 注意
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据，并且调整了batch_size等超参以便在样例数据上更加友好的显示训练&测试日志。如果需要复现readme中的效果请按照如下表格调整batch_size等超参，并使用提供的脚本下载对应数据集以及数据预处理。

|       模型       |       batch_size      |       thread_num      |       epoch_num      |
| :------------------: | :--------------------: | :--------------------: | :--------------------: |
|       Share-Bottom        |       32       |        1       |        400       |
|       MMoE        |       32       |       1       |        400       |
|       ESMM     |       64       |       2       |        100       |

### 数据处理
参考每个模型目录数据下载&预处理脚本

```
sh run.sh
```

### 训练
```
cd modles/multitask/mmoe # 进入选定好的排序模型的目录 以MMoE为例
python -m paddlerec.run -m ./config.yaml # 自定义修改超参后，指定配置文件，使用自定义配置
```

### 预测
```
# 修改对应模型的config.yaml, workspace配置为当前目录的绝对路径
# 修改对应模型的config.yaml，mode配置infer_runner
# 示例: mode: train_runner -> mode: infer_runner
# infer_runner中 class配置为 class: single_infer
# 修改phase阶段为infer的配置，参照config注释

# 修改完config.yaml后 执行:
python -m paddlerec.run -m ./config.yaml # 以MMoE为例
```


## 效果对比
### 模型效果列表

|       数据集        |       模型       |       loss        |       auc       | 
| :------------------: | :--------------------: | :---------: |:---------: |
|       Census-income Data     |       Share-Bottom       |       --        |     0.93120/0.99256         |
|       Census-income Data        |       MMoE       |       --        |       0.94465/0.99324         |
|          Ali-CCP     |    ESMM       |       --        |      0.97181/0.49967          |
