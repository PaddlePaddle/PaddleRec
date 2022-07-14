# kim (Personalized News Recommendation with Knowledge-aware Interactive Matching) 
代码请参考：[KIM](https://paddlerec.readthedocs.io/en/latest/models/match/kim.html) 
如果我们的代码对您有用，还请点个star啊~  


## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [模型组网](#模型组网)
- [效果复现](#效果复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)


## 数据准备
训练及测试数据集选用mind新闻、 glove.840B.300d 词向量初始化embedding层和知识图谱数据。

## 运行环境
PaddlePaddle>=2.0
nltk>=3.7
python 3.7

os : windows/linux/macos  

## 快速开始
**开始前确保已nltk英文分词模型到个人目录下**
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在kim模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/match/kim # 在任意目录均可运行
# 动态图训练
python -u trainer.py -m config.yaml -o mode=train # 全量数据运行config_bigdata.yaml 
# 动态图预测
python -u infer.py -m config.yaml -o mode=test
```  

## 模型组网
个性化新闻推荐的核心是候选新闻和用户兴趣之间的准确匹配。大多数现有的新闻推荐方法通常从文本内容中建立候选新闻模型，并从用户点击的新闻中建立用户兴趣模型，两者是独立的。然而，一篇新闻可能涵盖多个方面和实体，一个用户也可能有多种兴趣。对候选新闻和用户兴趣的独立建模可能会导致新闻和用户之间的劣质匹配。在本文中，我们提出了一个用于个性化新闻推荐的知识感知的交互式匹配框架。我们的方法可以对候选新闻和用户兴趣进行交互式建模，以学习用户感知的候选新闻表示和候选新闻感知的用户兴趣表示，这可以促进用户兴趣和候选新闻之间的准确匹配。更具体地说，我们提出了一个知识协同编码器，借助知识图谱捕捉实体中的关联性，为点击新闻和候选新闻交互式地学习基于知识的新闻表示。此外，我们还提出了一个文本协同编码器，通过对文本之间的语义关系进行建模，为被点击新闻和候选新闻交互式地学习基于文本的新闻表示。此外，我们还提出了一个用户-新闻联合编码器，从候选新闻和点击新闻的知识和基于文本的表征中学习候选新闻的用户兴趣表征和用户意识到的候选新闻表征，以实现更好的兴趣匹配。通过在两个真实世界的数据集上进行广泛的实验，我们证明了我们的方法可以有效地提高新闻推荐的性能。:
<p align="center">
<img align="center" src="../../../doc/imgs/kim.png">
<p>


## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。  
在全量数据下模型的指标如下：  
## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。  
在全量数据下模型的指标如下：  

| 模型  | AUC |  MRR   |    nDCG5 |   nDCG10  | batch_size | epoch_num | Time of each epoch |
|-----|-----|-----|-----|-----|------------|-----------|--------------------|
| kim |  0.6681   |   0.3164  |    0.3484 |  0.4132   | 16         | 7         | 2h                 |
|  kim   |   0.6696  |   0.3192  |   0.3515  |  0.4158   | 16         | 8         | 2h                 |

1. 确认您当前所在目录为PaddleRec/models/match/kim
2. 进入paddlerec/datasets/kim目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的kim全量数据集，并解压到指定文件夹。
``` bash
cd ../../../datasets/kim
bash run.sh
```
3. 切回模型目录
```bash
python -u trainer.py -m config_bigdata.yaml -o mode=train
python -u infer.py -m config_bigdata.yaml -o mode=test
```

## 进阶使用
  
## FAQ
