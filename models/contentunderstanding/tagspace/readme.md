# tagspace文本分类模型

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── train_data
        ├── small_train.csv #训练数据样例
    ├── test_data
        ├── small_test.csv #测试数据样例
    ├── text2paddle.py #数据处理程序
├── __init__.py
├── README.md #文档
├── model.py #模型文件
├── config.yaml #配置文件
├── reader.py #读取程序
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [效果复现](#效果复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)


## 模型简介
tagspace模型是一种对文本打标签的方法，它主要学习从短文到相关主题标签的映射。论文中主要利用CNN做doc向量， 然后优化 f(w,t+),f(w,t-)的距离作为目标函数，得到了 t（标签）和doc在一个特征空间的向量表达，这样就可以找 doc的hashtags了。  

论文[TAGSPACE: Semantic Embeddings from Hashtags](https://www.aclweb.org/anthology/D14-1194.pdf)中的网络结构如图所示，一层输入层，一个卷积层，一个pooling层以及最后一个全连接层进行降维。
<p align="center">
<img align="center" src="../../../doc/imgs/tagspace.png">
<p>

## 数据准备
[数据地址](https://github.com/mhjabreel/CharCNN/tree/master/data/) , [备份数据地址](https://paddle-tagspace.bj.bcebos.com/data.tar)

数据格式如下：  
```
"3","Wall St. Bears Claw Back Into the Black (Reuters)","Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again."
```

## 运行环境
PaddlePaddle>=1.7.2  

python 2.7/3.5/3.6/3.7  

PaddleRec >=0.1  

os : windows/linux/macos    


## 快速开始
本文提供了样例数据可以供您快速体验，在paddlerec目录下直接执行下面的命令即可启动训练： 

```
python -m paddlerec.run -m models/contentunderstanding/tagspace/config.yaml
```   


## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。  
1. 确认您当前所在目录为PaddleRec/models/contentunderstanding/tagspace  
2. 在data目录下载并解压数据集，命令如下：  
``` 
cd data
wget https://paddle-tagspace.bj.bcebos.com/data.tar
tar -xvf data.tar
```
3. 本文提供了快速将数据集中的汉字数据处理为可训练格式数据的脚本，您在解压数据集后，将原始数据存放在raw_big_train_data和raw_big_test_data两个目录下，并在python3环境下运行我们提供的text2paddle.py文件。即可生成可以直接用于训练的数据目录test_big_data和train_big_data。命令如下：
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

4. 退回tagspace目录中，打开文件config.yaml,更改其中的参数  
将workspace改为您当前的绝对路径。（可用pwd命令获取绝对路径）  
将dataset下sample_1的batch_size值从10改为128   
将dataset下sample_1的data_path改为：{workspace}/data/train_big_data  
将dataset下inferdata的batch_size值从10改为500 
将dataset下inferdata的data_path改为：{workspace}/data/test_big_data 

5.  执行命令，开始训练：
```
python -m paddlerec.run -m ./config.yaml
```
6. 运行结果：
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

## 动态图
```
# 进入模型目录
cd models/contentunderstanding/tagspace
# 训练
python -u train.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 预测
python -u infer.py -m config.yaml 
```
## 进阶使用
  
## FAQ
