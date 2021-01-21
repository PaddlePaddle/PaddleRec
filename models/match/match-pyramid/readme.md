# match-pyramid文本匹配模型

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── process.py #数据处理脚本
    ├── relation.test.fold1.txt #评估计算指标时用到的关系文件
    ├── train
    	├── train.txt #训练数据样例
    ├── test
    	├── test.txt #测试数据样例
├── __init__.py
├── README.md #文档
├── model.py #模型文件
├── config.yaml #配置文件
├── data_process.sh #数据下载和处理脚本
├── eval.py #计算指标的评估程序
├── run.sh #一键运行程序
├── test_reader.py #测试集读取程序
├── train_reader.py #训练集读取程序
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [论文复现](#论文复现)
- [动态图](#动态图)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)


## 模型简介
在许多自然语言处理任务中，匹配两个文本是一个基本问题。一种有效的方法是从单词，短语和句子中提取有意义的匹配模式以产生匹配分数。受卷积神经网络在图像识别中的成功启发，神经元可以根据提取的基本视觉模式（例如定向的边角和边角）捕获许多复杂的模式，所以我们尝试将文本匹配建模为图像识别问题。本模型对齐原作者庞亮开源的tensorflow代码：https://github.com/pl8787/MatchPyramid-TensorFlow/blob/master/model/model_mp.py， 实现了下述论文中提出的Match-Pyramid模型：

```text
@inproceedings{Pang L , Lan Y , Guo J , et al. Text Matching as Image Recognition[J]. 2016.,
  title={Text Matching as Image Recognition},
  author={Liang Pang, Yanyan Lan, Jiafeng Guo, Jun Xu, Shengxian Wan, Xueqi Cheng},
  year={2016}
}
```

## 数据准备
训练及测试数据集选用Letor07数据集和 embed_wiki-pdc_d50_norm 词向量初始化embedding层。  
该数据集包括：  
1.词典文件：我们将每个单词映射得到一个唯一的编号wid，并将此映射保存在单词词典文件中。例如：word_dict.txt  
2.语料库文件：我们使用字符串标识符的值表示一个句子的编号。第二个数字表示句子的长度。例如：qid_query.txt和docid_doc.txt  
3.关系文件：关系文件被用来存储两个句子之间的关系，如query 和document之间的关系。例如：relation.train.fold1.txt, relation.test.fold1.txt  
4.嵌入层文件：我们将预训练的词向量存储在嵌入文件中。例如：embed_wiki-pdc_d50_norm  

在本例中需要调用jieba库和sklearn库，如环境中没有提前安装，可以使用以下命令安装。  
```
pip install sklearn
pip install jieba
```

## 运行环境
PaddlePaddle>=1.7.2  
python 2.7/3.5/3.6/3.7  
PaddleRec >=0.1  
os : windows/linux/macos  

## 快速开始

本文提供了样例数据可以供您快速体验，在paddlerec目录下直接执行下面的命令即可启动训练： 

```
python -m paddlerec.run -m models/match/match-pyramid/config.yaml
```   

## 论文复现
1. 确认您当前所在目录为PaddleRec/models/match/match-pyramid
2. 本文提供了原数据集的下载以及一键生成训练和测试数据的预处理脚本，您可以直接一键运行:bash data_process.sh  
执行该脚本，会从国内源的服务器上下载Letor07数据集，并将完整的数据集解压到data文件夹。随后运行 process.py 将全量训练数据放置于`./data/big_train`，全量测试数据放置于`./data/big_test`。并生成用于初始化embedding层的embedding.npy文件  
执行该脚本的理想输出为：  
```
bash data_process.sh
...........load  data...............
--2020-07-13 13:24:50--  https://paddlerec.bj.bcebos.com/match_pyramid/match_pyramid_data.tar.gz
Resolving paddlerec.bj.bcebos.com... 10.70.0.165
Connecting to paddlerec.bj.bcebos.com|10.70.0.165|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 214449643 (205M) [application/x-gzip]
Saving to: “match_pyramid_data.tar.gz”

100%[==========================================================================================================>] 214,449,643  114M/s   in 1.8s

2020-07-13 13:24:52 (114 MB/s) - “match_pyramid_data.tar.gz” saved [214449643/214449643]

data/
data/relation.test.fold1.txt
data/relation.test.fold2.txt
data/relation.test.fold3.txt
data/relation.test.fold4.txt
data/relation.test.fold5.txt
data/relation.train.fold1.txt
data/relation.train.fold2.txt
data/relation.train.fold3.txt
data/relation.train.fold4.txt
data/relation.train.fold5.txt
data/relation.txt
data/docid_doc.txt
data/qid_query.txt
data/word_dict.txt
data/embed_wiki-pdc_d50_norm
...........data process...............
[./data/word_dict.txt]
        Word dict size: 193367
[./data/qid_query.txt]
        Data size: 1692
[./data/docid_doc.txt]
        Data size: 65323
[./data/embed_wiki-pdc_d50_norm]
        Embedding size: 109282
('Generate numpy embed:', (193368, 50))
[./data/relation.train.fold1.txt]
        Instance size: 47828
('Pair Instance Count:', 325439)
[./data/relation.test.fold1.txt]
        Instance size: 13652
```
3. 打开文件config.yaml,更改其中的参数  

将workspace改为您当前的绝对路径。（可用pwd命令获取绝对路径）
将dataset_train下的data_path参数改为{workspace}/data/big_train
将dataset_infer下的data_path参数改为{workspace}/data/big_test

4. 随后，您直接一键运行：bash run.sh  即可得到复现的论文效果
执行该脚本后，会执行python -m paddlerec.run -m ./config.yaml 命令开始训练并测试模型，将测试的结果保存到result.txt文件，最后通过执行eval.py进行评估得到数据的map指标  
执行该脚本的理想输出为：  
```
..............test.................
13651
336
('map=', 0.3993127885738651)
```  

## 动态图

在动态图中，训练和预测分离开，您需要在cofig.yaml以及config_bigdata.yaml中的dygraph部分配置动态图中需要的参数。  
```
# 进入模型目录
cd models/match/match-pyramid
# 训练
python -u train.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 预测
python -u infer.py -m config.yaml 
```
如需使用动态图进行效果复现，可以按以下步骤进行：
1. 在全量数据中执行训练时，需要将batch_size设置为128。  
2. 在全连数据中执行预测时，需要将batch_size设置为1，同时将print_interval设置为1.  
3. 将一键运行脚本run.sh的第一个命令：
“python -m paddlerec.run -m ./config_bigdata.yaml &> result.txt” 改为 “python -u infer.py -m config_bigdata.yaml &>result.txt”  
4. 执行sh run.sh
## 进阶使用
  
## FAQ
