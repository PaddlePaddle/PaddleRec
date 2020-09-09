# 内容理解模型库

## 简介
我们提供了常见的内容理解任务中使用的模型算法的PaddleRec实现, 单机训练&预测效果指标以及分布式训练&预测性能指标等。实现的内容理解模型包括 [Tagspace](tagspace)、[文本分类](textcnn)、[基于textcnn的预训练模型](textcnn_pretrain)等。

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
| TagSpace | 标签推荐 | [EMNLP 2014][TagSpace: Semantic Embeddings from Hashtags](https://www.aclweb.org/anthology/D14-1194.pdf) |
| textcnn | 文本分类 | [EMNLP 2014][Convolutional neural networks for sentence classication](https://www.aclweb.org/anthology/D14-1181.pdf) |

下面是每个模型的简介（注：图片引用自链接中的论文）

[TagSpace模型](https://www.aclweb.org/anthology/D14-1194.pdf)
<p align="center">
<img align="center" src="../../doc/imgs/tagspace.png">
<p>

[textCNN模型](https://www.aclweb.org/anthology/D14-1181.pdf)
<p align="center">
<img align="center" src="../../doc/imgs/cnn-ckim2014.png">
<p>

## 使用教程(快速开始)
```
git clone https://github.com/PaddlePaddle/PaddleRec.git paddle-rec
cd PaddleRec
python -m paddlerec.run -m models/contentunderstanding/tagspace/config.yaml
python -m paddlerec.run -m models/contentunderstanding/textcnn/config.yaml
```

## 使用教程（复现论文）

### 注意

为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果请使用以下提供的脚本下载对应数据集以及数据预处理。


**（1）TagSpace**

### 数据处理
[数据地址](https://github.com/mhjabreel/CharCNN/tree/master/data/) , [备份数据地址](https://paddle-tagspace.bj.bcebos.com/data.tar)

数据格式如下
```
"3","Wall St. Bears Claw Back Into the Black (Reuters)","Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again."
```

本文提供了快速将数据集中的汉字数据处理为可训练格式数据的脚本，您在解压数据集后，将原始数据存放在raw_big_train_data和raw_big_test_data两个目录下，并在python3环境下运行我们提供的text2paddle.py文件。即可生成可以直接用于训练的数据目录test_big_data和train_big_data。命令如下：
```
mkdir raw_big_train_data
mkdir raw_big_test_data
mv train.csv raw_big_train_data
mv test.csv raw_big_test_data
python3 text2paddle.py raw_big_train_data/ raw_big_test_data/ train_big_data test_big_data big_vocab_text.txt big_vocab_tag.txt
```

运行后的data目录：  

```
big_vocab_tag.txt  #标签词汇数
big_vocab_text.txt #文本词汇数
data.tar  #数据集
raw_big_train_data  #数据集中原始的训练集
raw_big_test_data  #数据集中原始的测试集
train_data  #样例训练集
test_data  #样例测试集
train_big_data  #数据集经处理后的训练集
test_big_data  #数据集经处理后的测试集
text2paddle.py  #预处理文件
```

处理完成的数据格式如下：
```
2,27 7062 8390 456 407 8 11589 3166 4 7278 31046 33 3898 2897 426 1
2,27 9493 836 355 20871 300 81 19 3 4125 9 449 462 13832 6 16570 1380 2874 5 0 797 236 19 3688 2106 14 8615 7 209 304 4 0 123 1
2,27 12754 637 106 3839 1532 66 0 379 6 0 1246 9 307 33 161 2 8100 36 0 350 123 101 74 181 0 6657 4 0 1222 17195 1
```


### 训练
退回tagspace目录中，打开文件config.yaml,更改其中的参数  
将workspace改为您当前的绝对路径。（可用pwd命令获取绝对路径）  
将dataset下sample_1的batch_size值从10改为128   
将dataset下sample_1的data_path改为：{workspace}/data/train_big_data  
将dataset下inferdata的batch_size值从10改为500 
将dataset下inferdata的data_path改为：{workspace}/data/test_big_data 
执行命令，开始训练：
```
python -m paddlerec.run -m ./config.yaml
```

### 预测
在跑完训练后，模型会开始在验证集上预测。
运行结果：
```
PaddleRec: Runner infer_runner Begin
Executor Mode: infer
processor_register begin
Running SingleInstance.
Running SingleNetwork.
Running SingleInferStartup.
Running SingleInferRunner.
load persistables from increment/9
batch: 1, acc: [0.91], loss: [0.02495437]
batch: 2, acc: [0.936], loss: [0.01941476]
batch: 3, acc: [0.918], loss: [0.02116447]
batch: 4, acc: [0.916], loss: [0.0219945]
batch: 5, acc: [0.902], loss: [0.02242816]
batch: 6, acc: [0.9], loss: [0.02421589]
batch: 7, acc: [0.9], loss: [0.026441]
batch: 8, acc: [0.934], loss: [0.01797657]
batch: 9, acc: [0.932], loss: [0.01687362]
batch: 10, acc: [0.926], loss: [0.02047823]
batch: 11, acc: [0.918], loss: [0.01998716]
batch: 12, acc: [0.898], loss: [0.0229556]
batch: 13, acc: [0.928], loss: [0.01736144]
batch: 14, acc: [0.93], loss: [0.01911209]
```

**（2）textcnn**

### 数据处理
情感倾向分析（Sentiment Classification，简称Senta）针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。情感类型分为积极、消极。情感倾向分析能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控，为企业提供有利的决策支持。  
情感是人类的一种高级智能行为，为了识别文本的情感倾向，需要深入的语义建模。另外，不同领域（如餐饮、体育）在情感的表达各不相同，因而需要有大规模覆盖各个领域的数据进行模型训练。为此，我们通过基于深度学习的语义模型和大规模数据挖掘解决上述两个问题。效果上，我们基于开源情感倾向分类数据集ChnSentiCorp进行评测。  
您可以直接执行以下命令下载我们分词完毕后的数据集,文件解压之后，senta_data目录下会存在训练数据（train.tsv）、开发集数据（dev.tsv）、测试集数据（test.tsv）以及对应的词典（word_dict.txt）：

``` 
wget https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
tar -zxvf sentiment_classification-dataset-1.0.0.tar.gz
```

数据格式为一句中文的评价语句，和一个代表情感信息的标签。两者之间用/t分隔，中文的评价语句已经分词，词之间用空格分隔。  

```
15.4寸 笔记本 的 键盘 确实 爽 ， 基本 跟 台式机 差不多 了 ， 蛮 喜欢 数字 小 键盘 ， 输 数字 特 方便 ， 样子 也 很 美观 ， 做工 也 相当 不错    1
跟 心灵 鸡汤 没 什么 本质 区别 嘛 ， 至少 我 不 喜欢 这样 读 经典 ， 把 经典 都 解读 成 这样 有点 去 中国 化 的 味道 了 0
```
本文提供了快速将数据集中的汉字数据处理为可训练格式数据的脚本，您在解压数据集后，将preprocess.py复制到senta_data文件中并执行，即可将数据集中提供的dev.tsv，test.tsv，train.tsv转化为可直接训练的dev.txt，test.txt，train.txt.
```
cp ./data/preprocess.py ./senta_data/
cd senta_data/
python preprocess.py
```

### 训练
创建存放训练集和测试集的目录，将数据放入目录中。
```
mkdir train
mv train.txt train
mkdir test
mv dev.txt  test
cd ..
```  

打开文件config.yaml,更改其中的参数  
将workspace改为您当前的绝对路径。（可用pwd命令获取绝对路径）  
将data1下的batch_size值从10改为128    
将data1下的data_path改为：{workspace}/senta_data/train  
将dataset_infer下的batch_size值从2改为256  
将dataset_infer下的data_path改为：{workspace}/senta_data/test  

执行命令，开始训练：
```
python -m paddlerec.run -m ./config.yaml
```

### 预测
在跑完训练后，模型会开始在验证集上预测。
运行结果：  

```
PaddleRec: Runner infer_runner Begin
Executor Mode: infer
processor_register begin
Running SingleInstance.
Running SingleNetwork.
Running SingleInferStartup.
Running SingleInferRunner.
load persistables from increment/14
batch: 1, acc: [0.91796875], loss: [0.2287855]
batch: 2, acc: [0.91796875], loss: [0.22827303]
batch: 3, acc: [0.90234375], loss: [0.27907994]
```


## 效果对比
### 模型效果 (测试)

|       数据集        |       模型       |       loss         |       acc         |
| :------------------: | :--------------------: | :---------: |:---------: | 
|       ag news dataset        |       TagSpace       |       0.0198        |       0.9177          | 
|       ChnSentiCorp        |       textcnn       |       0.2282        |        0.9127         | 
