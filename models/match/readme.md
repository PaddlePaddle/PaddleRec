# 匹配通用模型库

## 简介
我们提供了常见的匹配任务中均可以使用的模型算法的PaddleRec实现, 单机训练&预测效果指标以及分布式训练&预测性能指标等。实现的模型包括 [DSSM](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/match/dssm)、[MultiView-Simnet](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/match/multiview-simnet)。

模型算法库在持续添加中，欢迎关注。

## 目录
* [整体介绍](#整体介绍)
    * [匹配模型列表](#匹配模型列表)
* [使用教程](#使用教程)
    * [数据处理](#数据处理)
    * [训练](#训练)
    * [预测](#预测)
* [效果对比](#效果对比)
    * [模型效果列表](#模型效果列表)
* [分布式](#分布式)
    * [模型性能列表](#模型性能列表)

## 整体介绍
### 匹配模型列表

|       模型        |       简介        |       论文        |
| :------------------: | :--------------------: | :---------: |
| DSSM | Deep Structured Semantic Models | [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)(2013) |
| MultiView-Simnet | Multi-view Simnet for Personalized recommendation | [A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf)(2015) |

## 使用教程
### 数据处理
### 训练
### 预测

## 效果对比
### 模型效果列表

|       数据集        |       模型       |       loss        |       auc       | 
| :------------------: | :--------------------: | :---------: |:---------: |
|       -        |       DSSM       |       --        |       --          |
|       -        |       MultiView-Simnet       |       --        |       --          |

## 分布式
### 模型性能列表
|       数据集        |       模型       |       单机        |       多机（同步）          |       多机（异步）          |       GPU          |
| :------------------: | :--------------------: | :---------: |:---------: |:---------: |:---------: |
|       -        |       DSSM       |       --        |       --          |       --          |       --          |
|       -        |       MultiView-Simnet       |       --        |       --          |       --          |       --          |
