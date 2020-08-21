# multiview-simnet文本匹配模型

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
├── run.sh #运行脚本,在效果复现时使用
├── transform.py #整理格式准备计算指标的程序
├── reader.py #读者需要自定义数据集时供读者参考
├── evaluate_reader.py #读者需要自定义数据集时供读者参考
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
在个性化推荐场景中，推荐系统给用户提供的项目（Item）列表通常是通过个性化的匹配模型计算出来的。在现实世界中，一个用户可能有很多个视角的特征，比如用户Id，年龄，项目的点击历史等。一个项目，举例来说，新闻资讯，也会有多种视角的特征比如新闻标题，新闻类别等。多视角Simnet模型是可以融合用户以及推荐项目的多个视角的特征并进行个性化匹配学习的一体化模型。 多视角Simnet模型包括多个编码器模块，每个编码器被用在不同的特征视角上。当前，项目中提供Bag-of-Embedding编码器，Temporal-Convolutional编码器，和Gated-Recurrent-Unit编码器。我们会逐渐加入稀疏特征场景下比较实用的编码器到这个项目中。模型的训练方法，当前采用的是Pairwise ranking模式进行训练，即针对一对具有关联的User-Item组合，随机实用一个Item作为负例进行排序学习。 

模型的具体细节可以阅读论文[MultiView-Simnet](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf):
<p align="center">
<img align="center" src="../../../doc/imgs/multiview-simnet.png">
<p>

## 数据准备
我们公开了自建的测试集，包括百度知道、ECOM、QQSIM、UNICOM 四个数据集。这里我们选取百度知道数据集来进行训练。执行以下命令可以获取上述数据集。
```
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/simnet_dataset-1.0.0.tar.gz
tar xzf simnet_dataset-1.0.0.tar.gz
rm simnet_dataset-1.0.0.tar.gz
```

数据格式为一个标识句子的slot，后跟一个句子中词的token。两者形成{slot：token}的形式标识一个词：  
```
0:358 0:206 0:205 0:250 0:9 0:3 0:207 0:10 0:330 0:164 1:1144 1:217 1:206 1:9 1:3 1:207 1:10 1:398 1:2 2:217 2:206 2:9 2:3 2:207 2:10 2:398 2:2
0:358 0:206 0:205 0:250 0:9 0:3 0:207 0:10 0:330 0:164 1:951 1:952 1:206 1:9 1:3 1:207 1:10 1:398 2:217 2:206 2:9 2:3 2:207 2:10 2:398 2:2
```

## 运行环境
PaddlePaddle>=1.7.2  
python 2.7  
PaddleRec >=0.1  
os : linux  

## 快速开始
本文提供了样例数据可以供您快速体验，在paddlerec目录下直接执行下面的命令即可启动训练： 

```
python -m paddlerec.run -m models/match/multiview-simnet/config.yaml
```   


## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。  
1. 确认您当前所在目录为PaddleRec/models/match/multiview-simnet
2. 在data目录下载并解压数据集，命令如下：  
``` 
cd data
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/simnet_dataset-1.0.0.tar.gz
tar xzf simnet_dataset-1.0.0.tar.gz
rm -f simnet_dataset-1.0.0.tar.gz
mv data/zhidao ./
rm -rf data
```
3. 本文提供了快速将数据集中的汉字数据处理为可训练格式数据的脚本，您在解压数据集后，可以看见目录中存在一个名为zhidao的文件。然后能可以在python3环境下运行我们提供的preprocess.py文件。即可生成可以直接用于训练的数据目录test.txt,train.txt,label.txt和testquery.txt。将其放入train和test目录下以备训练时调用。命令如下：
```
python3 preprocess.py
rm -f ./train/train.txt
mv train.txt ./train
rm -f ./test/test.txt
mv test.txt ./test
cd ..
```
4. 退回tagspace目录中，打开文件config.yaml,更改其中的参数  

    将workspace改为您当前的绝对路径。（可用pwd命令获取绝对路径）  

5.  执行脚本，开始训练.脚本会运行python -m paddlerec.run -m ./config.yaml启动训练，并将结果输出到result文件中。然后启动格式整理程序transform，最后计算正逆序比：
```
sh run.sh
```

运行结果大致如下：
```
................run.................
!!! The CPU_NUM is not specified, you should set CPU_NUM in the environment variable list.
CPU_NUM indicates that how many CPUPlace are used in the current task.
And if this parameter are set as N (equal to the number of physical CPU core) the program may be faster.

export CPU_NUM=32 # for example, set CPU_NUM as number of physical CPU core which is 32.

!!! The default number of CPU_NUM=1.
I0821 14:24:57.255358  7888 parallel_executor.cc:440] The Program will be executed on CPU using ParallelExecutor, 1 cards are used, so 1 programs are executed in parallel.
I0821 14:24:57.259166  7888 build_strategy.cc:365] SeqOnlyAllReduceOps:0, num_trainers:1
I0821 14:24:57.262634  7888 parallel_executor.cc:307] Inplace strategy is enabled, when build_strategy.enable_inplace = True
I0821 14:24:57.264791  7888 parallel_executor.cc:375] Garbage collection strategy is enabled, when FLAGS_eager_delete_tensor_gb = 0
103
pnr: 1.17674418605
query_num: 11
pair_num: 468 468
equal_num: 0
正序率： 0.540598290598
253 215
```
6. 提醒：因为采取较小的数据集进行训练和测试，得到指标的浮动程度会比较大。如果得到的指标不合预期，可以多次执行步骤5，即可获得合理的指标。
## 进阶使用
  
## FAQ
