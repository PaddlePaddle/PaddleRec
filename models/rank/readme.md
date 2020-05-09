# 排序模型库

## 简介
我们提供了常见的排序任务中使用的模型算法，包括 [多层神经网络](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/rank/dnn)、[Deep Cross Network](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/rank/dcn)、[DeepFM](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/rank/deepfm)、 [xDeepFM](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/rank/xdeepfm)、[Deep Interest Network](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/rank/din)、[Wide&Deep](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/rank/wide_deep)。

模型算法库在持续添加中，欢迎关注。

## 目录
* [整体介绍](#整体介绍)
    * [排序模型列表](#排序模型列表)
* [使用教程](#使用教程)
    * [数据处理](#数据处理)
    * [训练](#训练)
    * [预测](#预测)
* [效果对比](#效果对比)
    * [模型效果列表](#模型效果列表)
* [分布式](#分布式)
    * [模型性能列表](#模型性能列表)

## 整体介绍
### 排序模型列表

|       模型        |       简介        |       论文        |
| :------------------: | :--------------------: | :---------: |
| DNN | 多层神经网络 | -- |
| wide&deep | Deep + wide(LR) | [Wide & Deep Learning for Recommender Systems](https://dl.acm.org/doi/abs/10.1145/2988450.2988454)(2016) |
| DeepFM | DeepFM | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)(2017) |
| xDeepFM | xDeepFM | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3219819.3220023)(2018) |
| DCN | Deep Cross Network | [Deep & Cross Network for Ad Click Predictions](https://dl.acm.org/doi/abs/10.1145/3124749.3124754)(2017) |
| DIN | Deep Interest Network | [Deep Interest Network for Click-Through Rate Prediction](https://dl.acm.org/doi/abs/10.1145/3219819.3219823)(2018) |

## 使用教程
### 数据处理
### 训练
### 预测

## 效果对比
### 模型效果列表

|       数据集        |       模型       |       loss        |       测试auc          |       acc         |       mae          |
| :------------------: | :--------------------: | :---------: |:---------: | :---------: |:---------: |
|       Criteo        |       DNN       |       --        |       0.79395          |       --          |       --          |
|       Criteo        |       DeepFM       |       0.44797        |       0.8046          |       --          |       --          |
|       Criteo        |       DCN       |       0.44703564        |       0.80654419          |       --          |       --          |
|       Criteo        |       xDeepFM       |       --        |       --          |       0.48657          |       --          |
|       Census-income Data        |       Wide&Deep       |       0.76195(mean)         |       0.90577(mean)          |       --          |       --          |
|       Amazon Product        |       DIN       |       0.47005194        |       0.863794952818         |       --          |       --          |

## 分布式
### 模型性能列表
|       数据集        |       模型       |       单机        |       多机（同步）          |       多机（异步）          |       GPU          |
| :------------------: | :--------------------: | :---------: |:---------: |:---------: |:---------: |
|       Criteo        |       DNN       |       --        |       --          |       --          |       --          |
|       Criteo        |       DeepFM       |       --        |       --          |       --          |       --          |
|       Criteo        |       DCN       |       --        |       --          |       --          |       --          |
|       Criteo        |       xDeepFM       |       --        |       --          |       --          |       --          |
|       Census-income Data        |       Wide&Deep       |       --        |       --          |       --          |       --          |
|       Amazon Product        |       DIN       |       --        |       --          |       --          |        --          |
