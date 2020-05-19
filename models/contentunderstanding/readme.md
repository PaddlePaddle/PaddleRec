# 内容理解模型库

## 简介
我们提供了常见的内容理解任务中使用的模型算法的PaddleRec实现, 单机训练&预测效果指标以及分布式训练&预测性能指标等。实现的内容理解模型包括 [Tagspace](tagspace)、[文本分类](classification)等。

模型算法库在持续添加中，欢迎关注。

## 目录
* [整体介绍](#整体介绍)
    * [模型列表](#内容理解模型列表)
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
| TagSpace | 标签推荐 | [TagSpace: Semantic Embeddings from Hashtags (2014)](https://research.fb.com/publications/tagspace-semantic-embeddings-from-hashtags/) |
| Classification | 文本分类 | [Convolutional neural networks for sentence classication (2014)](https://www.aclweb.org/anthology/D14-1181.pdf) |

下面是每个模型的简介（注：图片引用自链接中的论文）

[TagSpace模型](https://research.fb.com/publications/tagspace-semantic-embeddings-from-hashtags)
<p align="center">
<img align="center" src="../../doc/imgs/tagspace.png">
<p>

[文本分类CNN模型](https://www.aclweb.org/anthology/D14-1181.pdf)
<p align="center">
<img align="center" src="../../doc/imgs/cnn-ckim2014.png">
<p>

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

**（2）Classification**

无

### 训练

```
python -m paddlerec.run -m paddlerec.models.contentunderstanding.classification
```

### 预测

```
python -m paddlerec.run -m paddlerec.models.contentunderstanding.classification
```

## 效果对比
### 模型效果 (测试)

|       数据集        |       模型       |       loss        |       auc          |       acc         |       mae          |
| :------------------: | :--------------------: | :---------: |:---------: | :---------: |:---------: |
|       ag news dataset        |       TagSpace       |       --        |       --          |       --          |       --          |
|       --        |       Classification       |       --        |       --          |       --          |       --          |

