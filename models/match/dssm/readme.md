# DSSM文本匹配模型

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
├── synthetic_reader.py #读取训练集的程序
├── synthetic_evaluate_reader.py #读取测试集的程序
├── eval.py #评价脚本
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
DSSM是Deep Structured Semantic Model的缩写，即我们通常说的基于深度网络的语义模型，其核心思想是将query和doc映射到到共同维度的语义空间中，通过最大化query和doc语义向量之间的余弦相似度，从而训练得到隐含语义模型，达到检索的目的。DSSM有很广泛的应用，比如：搜索引擎检索，广告相关性，问答系统，机器翻译等。    
DSSM 的输入采用 BOW（Bag of words）的方式，相当于把字向量的位置信息抛弃了，整个句子里的词都放在一个袋子里了。将一个句子用这种方式转化为一个向量输入DNN中。  
Query 和 Doc 的语义相似性可以用这两个向量的 cosine 距离表示，然后通过softmax 函数选出与Query语义最相似的样本 Doc 。  

模型的具体细节可以阅读论文[DSSM](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf):
<p align="center">
<img align="center" src="../../../doc/imgs/dssm.png">
<p>

## 数据准备
我们公开了自建的测试集，包括百度知道、ECOM、QQSIM、UNICOM 四个数据集。这里我们选取百度知道数据集来进行训练。执行以下命令可以获取上述数据集。
```
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/simnet_dataset-1.0.0.tar.gz
tar xzf simnet_dataset-1.0.0.tar.gz
rm simnet_dataset-1.0.0.tar.gz
```

## 运行环境
PaddlePaddle>=1.7.2

python 2.7/3.5/3.6/3.7

PaddleRec >=0.1

os : windows/linux/macos

## 快速开始
本文提供了样例数据可以供您快速体验，直接执行下面的命令即可启动训练： 

```
python -m paddlerec.run -m models/match/dssm/config.yaml
```   


## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。  
1. 确认您当前所在目录为PaddleRec/models/match/dssm
2. 在data目录下载并解压数据集，命令如下：  
``` 
cd data
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/simnet_dataset-1.0.0.tar.gz
tar xzf simnet_dataset-1.0.0.tar.gz
rm simnet_dataset-1.0.0.tar.gz
```
3. 本文提供了快速将数据集中的汉字数据处理为可训练格式数据的脚本，您在解压数据集后，可以看见目录中存在一个名为zhidao的文件。然后能可以在python3环境下运行我们提供的preprocess.py文件。即可生成可以直接用于训练的数据目录test.txt,train.txt和label.txt。将其放入train和test目录下以备训练时调用。命令如下：
```
mv data/zhidao ./
rm -rf data
python3 preprocess.py
rm -f ./train/train.txt
mv train.txt ./train
rm -f ./test/test.txt
mv test.txt test
cd ..
```
经过预处理的格式：  
训练集为三个稀疏的BOW方式的向量：query,pos,neg  
测试集为两个稀疏的BOW方式的向量：query,pos  
label.txt中对应的测试集中的标签

4. 退回tagspace目录中，打开文件config.yaml,更改其中的参数  

将workspace改为您当前的绝对路径。（可用pwd命令获取绝对路径）  
将dataset_train中的batch_size从8改为128

5.  执行脚本，开始训练.脚本会运行python -m paddlerec.run -m ./config.yaml启动训练，并将结果输出到result文件中。然后启动评价脚本eval.py计算auc：
```
sh run.sh
```

## 进阶使用
  
## FAQ
