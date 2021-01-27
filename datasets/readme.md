# 推荐系统数据集
这是PaddleRec的数据集的的存储库。您可以在这里方便的一键下载我们处理完成的数据集，也可以使用PaddleRec轻松测试这些数据集上不同推荐模型的性能。

## 数据集的使用方法
在PaddleRec/datasets目录下，您可以看到很多放置数据集的子目录，每个目录下都有一个`run.sh`脚本。执行下面的命令运行脚本即可一键下载预处理完成的数据集。  
```bash
cd xxx      # xxx为您需要下载的数据集目录名
sh run.sh
```
同时，目录下也会有一个data_process.sh脚本，可以供您自己处理数据集。
```bash
cd xxx      # xxx为您需要下载的数据集目录名
sh data_process.sh
```

## 数据集简介
 |                    数据集名称                    |                                           简介                                           |                 Reference                 |
 | :----------------------------------------------: | :------------------------------------------------------------------------------------------: | :-------------------------------: |
 |[ag_news](https://paddle-tagspace.bj.bcebos.com/data.tar)|496835 条来自AG新闻语料库 4 大类别超过 2000 个新闻源的新闻文章，数据集仅仅援用了标题和描述字段。每个类别分别拥有 30,000 个训练样本及 1900 个测试样本。| [ComeToMyHead](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)|
 |[Ali-CCP：Alibaba Click and Conversion Prediction](  https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408  )|从淘宝推荐系统的真实流量日志中收集的数据集。|[SIGIR(2018)]( https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408)|
 |[BQ](https://paddlerec.bj.bcebos.com/dssm%2Fbq.tar.gz)|BQ是一个智能客服中文问句匹配数据集，该数据集是自动问答系统语料，共有120,000对句子对，并标注了句子对相似度值。数据中存在错别字、语法不规范等问题，但更加贴近工业场景|--|
 |[Census-income Data](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz )|此数据集包含从1994年和1995年美国人口普查局进行的当前人口调查中提取的加权人口普查数据。数据包含人口统计和就业相关变量。|[Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid](http://robotics.stanford.edu/~ronnyk/nbtree.pdf)|
 |[Criteo](https://fleet.bj.bcebos.com/ctr_data.tar.gz)|该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。|[kaggle](https://www.kaggle.com/c/criteo-display-ad-challenge/)|
 |[letor07](https://paddlerec.bj.bcebos.com/match_pyramid/match_pyramid_data.tar.gz)|LETOR是一套用于学习排名研究的基准数据集，其中包含标准特征、相关性判断、数据划分、评估工具和若干基线|[LETOR: Learning to Rank for Information Retrieval](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fbeijing%2Fprojects%2Fletor%2F)|
 |[senti_clas](https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz)|情感倾向分析（Sentiment Classification，简称Senta）针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。情感类型分为积极、消极。情感倾向分析能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控，为企业提供有利的决策支持|--|
 |[one_billion](http://www.statmt.org/lm-benchmark/)|拥有十亿个单词基准，为语言建模实验提供标准的训练和测试|[One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling](https://arxiv.org/abs/1312.3005)|
