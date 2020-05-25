# 重排序模型库

## 简介
我们提供了常见的重排序使用的模型算法的PaddleRec实现, 单机训练&预测效果指标以及分布式训练&预测性能指标等。目前实现的模型是 [Listwise](listwise)。

模型算法库在持续添加中，欢迎关注。

## 目录
* [整体介绍](#整体介绍)
    * [重排序模型列表](#重排序模型列表)
* [使用教程](#使用教程)
    * [训练 预测](#训练 预测)
* [效果对比](#效果对比)
    * [模型效果列表](#模型效果列表)

## 整体介绍
### 融合模型列表

|       模型        |       简介        |       论文        |
| :------------------: | :--------------------: | :---------: |
| Listwise | Listwise | [Sequential Evaluation and Generation Framework for Combinatorial Recommender System](https://arxiv.org/pdf/1902.00245.pdf)(2019) |

下面是每个模型的简介（注：图片引用自链接中的论文）


[Listwise](https://arxiv.org/pdf/1902.00245.pdf):
<p align="center">
<img align="center" src="../../doc/imgs/listwise.png">
<p>


## 使用教程
### 训练 预测
```shell
python -m paddlerec.run -m paddlerec.models.rerank.listwise # listwise
```

## 效果对比
### 模型效果列表

|       数据集        |       模型       |       loss        |       auc       | 
| :------------------: | :--------------------: | :---------: |:---------: |
|    --     |       Listwise       |       --        |     --         |
