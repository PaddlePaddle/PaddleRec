# 内容理解模型库

## 简介
我们提供了常见的内容理解任务中使用的模型算法的PaddleRec实现, 单机训练&预测效果指标以及分布式训练&预测性能指标等。实现的内容理解模型包括 [Tagspace](http://gitlab.baidu.com/xujiaqi01/paddlerec/tree/develop/models/contentunderstanding/tagspace)、[文本分类](http://gitlab.baidu.com/xujiaqi01/paddlerec/tree/develop/models/contentunderstanding/text_classification)。

模型算法库在持续添加中，欢迎关注。

## 目录
* [整体介绍](#整体介绍)
    * [内容理解模型列表](#内容理解模型列表)
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
| TagSpace | 标签推荐 | [TagSpace: Semantic Embeddings from Hashtags](https://research.fb.com/publications/tagspace-semantic-embeddings-from-hashtags/) |
| TextClassification | 文本分类 | -- |


## 使用教程
### 数据处理

**（1）TagSpace**

[数据地址](https://github.com/mhjabreel/CharCNN/tree/master/data/) , [备份数据地址](https://paddle-tagspace.bj.bcebos.com/data.tar)
 
数据格式如下
```
"3","Wall St. Bears Claw Back Into the Black (Reuters)","Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again."
```

数据解压后，将文本数据转为paddle数据，先将数据放到训练数据目录和测试数据目录

```
mkdir raw_big_train_data
mkdir raw_big_test_data
mv train.csv raw_big_train_data
mv test.csv raw_big_test_data
```

运行脚本text2paddle.py 生成paddle输入格式

```
python text2paddle.py raw_big_train_data/ raw_big_test_data/ train_big_data test_big_data big_vocab_text.txt big_vocab_tag.txt
```

**（2）TextClassification**

无

### 训练
### 预测

## 效果对比
### 模型效果 (测试)

|       数据集        |       模型       |       loss        |       auc          |       acc         |       mae          |
| :------------------: | :--------------------: | :---------: |:---------: | :---------: |:---------: |
|       --        |       TagSpace       |       --        |       --          |       --          |       --          |
|       --        |       TextClassification       |       --        |       --          |       --          |       --          |


## 分布式
### 模型训练性能 (样本/s)
|       数据集        |       模型       |       单机        |       同步 (4节点)          |       同步 (8节点)          |  同步 (16节点)          |  同步 (32节点)          |
| :------------------: | :--------------------: | :---------: |:---------: |:---------: |:---------: |:---------: |
|       --        |       TagSpace       |       --        |       --          |       --          |  --          |  --          |
|       --        |       TextClassification       |       --        |       --          |       --          |   --          |   --          |


----

|       数据集        |       模型       |       单机        |       异步 (4节点)          |       异步 (8节点)          |  异步 (16节点)          |  异步 (32节点)          |
| :------------------: | :--------------------: | :---------: |:---------: |:---------: |:---------: |:---------: |
|       --        |       TagSpace       |       --        |       --          |       --          |  --          |  --          |
|       --        |       TextClassification       |       --        |       --          |       --          |   --          |   --          |