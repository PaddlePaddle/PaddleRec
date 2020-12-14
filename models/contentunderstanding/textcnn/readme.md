# textcnn文本分类模型

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── train
        ├── train.txt #训练数据样例
    ├── test
        ├── test.txt #测试数据样例
    ├── preprocess.py #数据处理程序
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
TextCNN网络是2014年提出的用来做文本分类的卷积神经网络，由于其结构简单、效果好，在文本分类、推荐等NLP领域应用广泛。对于文本分类问题，常见的方法无非就是抽取文本的特征。然后再基于抽取的特征训练一个分类器。 然而研究证明，TextCnn在文本分类问题上有着更加卓越的表现。从直观上理解，TextCNN通过一维卷积来获取句子中n-gram的特征表示。TextCNN对文本浅层特征的抽取能力很强，在短文本领域专注于意图分类时效果很好，应用广泛，且速度较快。  
Yoon Kim在论文[EMNLP 2014][Convolutional neural networks for sentence classication](https://www.aclweb.org/anthology/D14-1181.pdf)提出了TextCNN并给出基本的结构。将卷积神经网络CNN应用到文本分类任务，利用多个不同size的kernel来提取句子中的关键信息（类似于多窗口大小的ngram），从而能够更好地捕捉局部相关性。模型的主体结构如图所示：  
<p align="center">
<img align="center" src="../../../doc/imgs/cnn-ckim2014.png">
<p>

## 数据准备
情感倾向分析（Sentiment Classification，简称Senta）针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。情感类型分为积极、消极。情感倾向分析能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控，为企业提供有利的决策支持。  
情感是人类的一种高级智能行为，为了识别文本的情感倾向，需要深入的语义建模。另外，不同领域（如餐饮、体育）在情感的表达各不相同，因而需要有大规模覆盖各个领域的数据进行模型训练。为此，我们通过基于深度学习的语义模型和大规模数据挖掘解决上述两个问题。效果上，我们基于开源情感倾向分类数据集ChnSentiCorp进行评测，模型在测试集上的准确率如表所示：  

| 模型 | dev | test | 
| :------| :------ | :------
| TextCNN | 90.75% | 91.27% |


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

## 运行环境
PaddlePaddle>=1.7.2

python 2.7/3.5/3.6/3.7

PaddleRec >=0.1

os : windows/linux/macos


## 快速开始
本文提供了样例数据可以供您快速体验，在paddlerec目录下直接执行下面的命令即可启动训练： 

```
python -m paddlerec.run -m models/contentunderstanding/textcnn/config.yaml
```   


## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。  
1. 确认您当前所在目录为PaddleRec/models/contentunderstanding/textcnn  
2. 下载并解压数据集，命令如下：  
``` 
wget https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
tar -zxvf sentiment_classification-dataset-1.0.0.tar.gz
```
3. 本文提供了快速将数据集中的汉字数据处理为可训练格式数据的脚本，您在解压数据集后，将preprocess.py复制到senta_data文件中并执行，即可将数据集中提供的dev.tsv，test.tsv，train.tsv转化为可直接训练的dev.txt，test.txt，train.txt.
```
cp ./data/preprocess.py ./senta_data/
cd senta_data/
python preprocess.py
```
4. 创建存放训练集和测试集的目录，将数据放入目录中。
```
mkdir train
mv train.txt train
mkdir test
mv dev.txt  test
cd ..
```  
5. 打开文件config.yaml,更改其中的参数  

将workspace改为您当前的绝对路径。（可用pwd命令获取绝对路径）  
将data1下的batch_size值从10改为128    
将data1下的data_path改为：{workspace}/senta_data/train  
将dataset_infer下的batch_size值从2改为256  
将dataset_infer下的data_path改为：{workspace}/senta_data/test  

6.  执行命令，开始训练：
```
python -m paddlerec.run -m ./config.yaml
```

7. 运行结果：
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
## 进阶使用

### 动态图
```
# 进入模型目录
cd models/contentunderstanding/textcnn
# 训练
python -u train.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 预测
python -u infer.py -m config.yaml 
```
  
## FAQ
