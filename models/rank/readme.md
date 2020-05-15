# 排序模型库

## 简介
我们提供了常见的排序任务中使用的模型算法的PaddleRec实现, 单机训练&预测效果指标以及分布式训练&预测性能指标等。实现的排序模型包括 [多层神经网络](dnn)、[Deep Cross Network](dcn)、[DeepFM](deepfm)、 [xDeepFM](xdeepfm)、[Deep Interest Network](din)、[Wide&Deep](wide_deep)。

模型算法库在持续添加中，欢迎关注。

## 目录
* [整体介绍](#整体介绍)
    * [模型列表](#模型列表)
* [使用教程](#使用教程)
    * [数据处理](#数据处理)
    * [训练](#训练)
    * [预测](#预测)
* [效果对比](#效果对比)
    * [模型效果列表](#模型效果列表)
* [分布式](#分布式)
    * [模型性能列表](#模型性能列表)

## 整体介绍
### 模型列表

|       模型        |       简介        |       论文        |
| :------------------: | :--------------------: | :---------: |
| DNN | 多层神经网络 | -- |
| wide&deep | Deep + wide(LR) | [Wide & Deep Learning for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454)(2016) |
| DeepFM | DeepFM | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)(2017) |
| DCN | Deep Cross Network | [Deep & Cross Network for Ad Click Predictions](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754)(2017) |
| xDeepFM | xDeepFM | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/3219819.3220023)(2018) |
| DIN | Deep Interest Network | [Deep Interest Network for Click-Through Rate Prediction](https://dl.acm.org/doi/pdf/10.1145/3219819.3219823)(2018) |

[wide&deep](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454):
<p align="center">
<img align="center" src="../../doc/imgs/wide&deep.jpg">
<p>

[DeepFM](https://arxiv.org/pdf/1703.04247.pdf):
<p align="center">
<img align="center" src="../../doc/imgs/deepfm.jpg">
<p>

[XDeepFM](https://dl.acm.org/doi/pdf/10.1145/3219819.3220023):
<p align="center">
<img align="center" src="../../doc/imgs/xdeepfm.jpg">
<p>

[DCN](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754):
<p align="center">
<img align="center" src="../../doc/imgs/dcn.jpg">
<p>

[DIN](https://dl.acm.org/doi/pdf/10.1145/3219819.3219823):
<p align="center">
<img align="center" src="../../doc/imgs/din.jpg">
<p>

## 使用教程
### 数据处理
### 训练
### 预测

## 效果对比
### 模型效果 (测试)

|       数据集        |       模型       |       loss        |       auc          |       acc         |       mae          |
| :------------------: | :--------------------: | :---------: |:---------: | :---------: |:---------: |
|       Criteo        |       DNN       |       --        |       0.79395          |       --          |       --          |
|       Criteo        |       DeepFM       |       0.44797        |       0.80460          |       --          |       --          |
|       Criteo        |       DCN       |       0.44704        |       0.80654          |       --          |       --          |
|       Criteo        |       xDeepFM       |       --        |       --          |       0.48657          |       --          |
|       Census-income Data        |       Wide&Deep       |       0.76195         |       0.90577          |       --          |       --          |
|       Amazon Product        |       DIN       |       0.47005        |       0.86379         |       --          |       --          |

## 分布式
### 模型训练性能 (样本/s)
|       数据集        |       模型       |       单机        |       同步 (4节点)          |       同步 (8节点)          |  同步 (16节点)          |  同步 (32节点)          |
| :------------------: | :--------------------: | :---------: |:---------: |:---------: |:---------: |:---------: |
|       Criteo        |       DNN       |       99821        |       148788          |       148788          |  507936          |  856032          |
|       Criteo        |       DeepFM       |       --        |       --          |       --          |   --          |   --          |
|       Criteo        |       DCN       |       --        |       --          |       --          |  --          |  --          |
|       Criteo        |       xDeepFM       |       --        |       --          |       --          |  --          |  --          |
|       Census-income Data        |       Wide&Deep       |       --        |       --          |       --          |  --          |  --          |
|       Amazon Product        |       DIN       |       --        |       --          |       --          |  --          |  --          |

----

|       数据集        |       模型       |       单机        |       异步 (4节点)          |       异步 (8节点)          |  异步 (16节点)          |  异步 (32节点)          |
| :------------------: | :--------------------: | :---------: |:---------: |:---------: |:---------: |:---------: |
|       Criteo        |       DNN       |       99821        |       316918          |       602536          |  1130557          |  2048384          |
|       Criteo        |       DeepFM       |       --        |       --          |       --          |   --          |   --          |
|       Criteo        |       DCN       |       --        |       --          |       --          |  --          |  --          |
|       Criteo        |       xDeepFM       |       --        |       --          |       --          |  --          |  --          |
|       Census-income Data        |       Wide&Deep       |       --        |       --          |       --          |  --          |  --          |
|       Amazon Product        |       DIN       |       --        |       --          |       --          |  --          |  --          |
