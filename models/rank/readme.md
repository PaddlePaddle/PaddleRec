# Rank模型库

## 简介
我们提供了常见的ctr任务中使用的模型，包括 [dnn](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/rank/dnn)、[dcn](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/rank/dcn)、[deepfm](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/rank/deepfm)、 [xdeepfm](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/rank/xdeepfm)、[din](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/rank/din)、[wide&deep](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/rank/wide_deep)。

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
| wide&deep | Deep + wide(LR) | [论文链接](https://dl.acm.org/doi/abs/10.1145/2988450.2988454)(2016) |
| DeepFM | Deep + FM 并行 | [论文链接](https://arxiv.org/abs/1703.04247)(2017) |
| xDeepFM | DeepFM升级版 | [论文链接](https://dl.acm.org/doi/abs/10.1145/3219819.3220023)(2018) |
| DCN | wide升级为Cross Layer Network | [论文链接](https://dl.acm.org/doi/abs/10.1145/3124749.3124754)(2017) |
| DIN | Embeddding层引入attention机制 | [论文链接](https://dl.acm.org/doi/abs/10.1145/3219819.3219823)(2018) |

## 使用教程
### 数据处理
### 训练
### 预测

## 效果对比
### 模型效果列表

|       数据集        |       模型       |       单机测试集指标        |       详情          |
| :------------------: | :--------------------: | :---------: |:---------: |
|       Criteo        |       DNN       |       auc：0.79395        |       [更多](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/dnn#benchmark)          |
|       Criteo        |       DeepFM       |       logloss: 0.44797, <br>auc：0.8046        |       [更多](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/deepfm#result)          |
|       Criteo        |       DCN       |       logloss: 0.44703564, <br>auc: 0.80654419        |       [更多](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/dcn#%E7%BB%93%E6%9E%9C)          |
|       Demo数据集        |       xDeepFM       |       acc: 0.48657, <br>auc：0.7308        |       [更多](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/xdeepfm#%E5%8D%95%E6%9C%BA%E7%BB%93%E6%9E%9C)          |
|       Census-income Data        |       Wide&Deep       |       mean_acc:0.76195, <br>mean_auc:0.90577         |       [更多](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/wide_deep#%E6%A8%A1%E5%9E%8B%E6%95%88%E6%9E%9C)          |
|       Amazon Product        |       DIN       |       logloss: 0.47005194, <br>auc: 0.863794952818        |       [更多](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr/din#%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C%E7%A4%BA%E4%BE%8B)          |

## 分布式
### 模型性能列表
|       数据集        |       模型       |       单机        |       多机（同步）          |       多机（异步）          |
| :------------------: | :--------------------: | :---------: |:---------: |:---------: |
|       Criteo        |       DNN       |       --        |       --          |       --          |
|       Criteo        |       DeepFM       |       --        |       --          |       --          |
|       Criteo        |       DCN       |       --        |       --          |       --          |
|       Demo数据集        |       xDeepFM       |       --        |       --          |       --          |
|       Census-income Data        |       Wide&Deep       |       --        |       --          |       --          |
|       Amazon Product        |       DIN       |       --        |       --          |       --          | 
