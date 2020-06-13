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
| TagSpace | 标签推荐 | [EMNLP 2014][TagSpace: Semantic Embeddings from Hashtags](https://research.fb.com/publications/tagspace-semantic-embeddings-from-hashtags/) |
| Classification | 文本分类 | [EMNLP 2014][Convolutional neural networks for sentence classication](https://www.aclweb.org/anthology/D14-1181.pdf) |

下面是每个模型的简介（注：图片引用自链接中的论文）

[TagSpace模型](https://research.fb.com/publications/tagspace-semantic-embeddings-from-hashtags)
<p align="center">
<img align="center" src="../../doc/imgs/tagspace.png">
<p>

[文本分类CNN模型](https://www.aclweb.org/anthology/D14-1181.pdf)
<p align="center">
<img align="center" src="../../doc/imgs/cnn-ckim2014.png">
<p>

##使用教程(快速开始)
```
python -m paddlerec.run -m paddlerec.models.contentunderstanding.tagspace
python -m paddlerec.run -m paddlerec.models.contentunderstanding.classification
```

## 使用教程（复现论文）

###注意

为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果请使用以下提供的脚本下载对应数据集以及数据预处理。

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

### 训练
```
cd modles/contentunderstanding/tagspace
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
python -m paddlerec.run -m ./config.yaml
```

**（2）Classification**

### 训练
```
cd modles/contentunderstanding/classification
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
python -m paddlerec.run -m ./config.yaml
```

## 效果对比
### 模型效果 (测试)

|       数据集        |       模型       |       loss        |       auc          |       acc         |       mae          |
| :------------------: | :--------------------: | :---------: |:---------: | :---------: |:---------: |
|       ag news dataset        |       TagSpace       |       --        |       --          |       --          |       --          |
|       --        |       Classification       |       --        |       --          |       --          |       --          |
