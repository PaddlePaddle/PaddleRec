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
 |                    数据集名称                    |     模型reader示例     |                                           简介                                           |                 Reference                 |
 | :----------------------------------------------: |:----------------------------:| :------------------------------------------------------------------------------------------: | :-------------------------------: |
 |[ag_news](https://paddle-tagspace.bj.bcebos.com/data.tar)| [tagspace](../models/contentunderstanding/tagspace/agnews_reader.py)|496835 条来自AG新闻语料库 4 大类别超过 2000 个新闻源的新闻文章，数据集仅仅援用了标题和描述字段。每个类别分别拥有 30,000 个训练样本及 1900 个测试样本。| [ComeToMyHead](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)|
 |[Ali-CCP：Alibaba Click and Conversion Prediction](https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408)| [esmm](../models/multitask/esmm/esmm_reader.py) |从淘宝推荐系统的真实流量日志中收集的数据集。|[SIGIR(2018)]( https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408)|
 |[BQ](https://paddlerec.bj.bcebos.com/dssm%2Fbq.tar.gz)| [dssm](../models/match/dssm/bq_reader_train.py) |BQ是一个智能客服中文问句匹配数据集，该数据集是自动问答系统语料，共有120,000对句子对，并标注了句子对相似度值。数据中存在错别字、语法不规范等问题，但更加贴近工业场景|[The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification](http://icrc.hitsz.edu.cn/info/1037/1162.htm)|
 |[Census-income Data](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz )| [ple](../models/multitask/ple/census_reader.py) |此数据集包含从1994年和1995年美国人口普查局进行的当前人口调查中提取的加权人口普查数据。数据包含人口统计和就业相关变量。|[Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid](http://robotics.stanford.edu/~ronnyk/nbtree.pdf)|
 |[Criteo](https://fleet.bj.bcebos.com/ctr_data.tar.gz)| [wide_deep](../models/rank/wide_deep/criteo_reader.py) |该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。|[kaggle](https://www.kaggle.com/c/criteo-display-ad-challenge/)|
 |[letor07](https://paddlerec.bj.bcebos.com/match_pyramid/match_pyramid_data.tar.gz)| [match-pyramid](../models/match/match-pyramid/letor_reader.py) |LETOR是一套用于学习排名研究的基准数据集，其中包含标准特征、相关性判断、数据划分、评估工具和若干基线|[LETOR: Learning to Rank for Information Retrieval](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fbeijing%2Fprojects%2Fletor%2F)|
 |[senti_clas](https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz)| [textcnn](../models/contentunderstanding/textcnn/senti_clas_reader.py)|情感倾向分析（Sentiment Classification，简称Senta）针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。情感类型分为积极、消极。情感倾向分析能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控，为企业提供有利的决策支持|--|
 |[one_billion](http://www.statmt.org/lm-benchmark/)| [word2vec](../models/recall/word2vec/word2vec_reader.py) |拥有十亿个单词基准，为语言建模实验提供标准的训练和测试|[One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling](https://arxiv.org/abs/1312.3005)|
 |[MIND](https://paddlerec.bj.bcebos.com/datasets/MIND/bigdata.zip)| [naml](../models/rank/naml/NAMLDataReader.py) |MIND即MIcrosoft News Dataset的简写，MIND里的数据来自Microsoft News用户的行为日志。MIND的数据集里包含了1,000,000的用户以及这些用户与160,000的文章的交互行为。|[Microsoft(2020)](https://msnews.github.io)|
 |[movielens_pinterest_NCF](https://paddlerec.bj.bcebos.com/ncf/Data.zip)| [NCF](../models/recall/ncf/movielens_reader.py) |论文原作者处理过的movielens数据集和pinterest数据集，[github](https://github.com/hexiangnan/neural_collaborative_filtering)|[《Neural Collaborative Filtering 》](https://arxiv.org/pdf/1708.05031.pdf)|
 |[Anime](https://paddlerec.bj.bcebos.com/datasets/Anime/archive.zip)| -- |该数据集包含73,516个用户对12,294个动漫的用户偏好数据。每个用户都可以将动漫添加到列表中并给它一个评分，该数据集是这些评分的汇总。|[Kaggle](https://www.kaggle.com/CooperUnion/anime-recommendations-database)|
 |[LFM-1b](https://paddlerec.bj.bcebos.com/datasets/LFM_1b/LFM-1b.zip)| -- |此数据集包含由Last.FM的120,000多个用户创建的十亿多个音乐收听记录。每条收听记录均以艺术家，专辑和曲目名称为特征，并包含一个时间戳。|[ICMR 2016](http://www.cp.jku.at/datasets/LFM-1b/)|
 |[LFM-1b UGP](https://paddlerec.bj.bcebos.com/datasets/LFM_1b_UGP/LFM-1b_UGP.zip)| -- |LFM-1b数据集的用户类型档案,作为LFM-1b的补充扩展|[ISM 2017](http://www.cp.jku.at/datasets/LFM-1b/)|
 |[Jester](https://paddlerec.bj.bcebos.com/datasets/Jester/JesterDataset3.zip)| -- |此数据集包含Jester Joke Recommender系统用户对笑话的匿名评分。|[UC Berkeley](http://eigentaste.berkeley.edu/dataset/)|
 |[Steam](https://paddlerec.bj.bcebos.com/datasets/steam/steam_reviews.json.gz)| -- |该数据集是Steam的评论和游戏信息，其中包含7,793,069条评论，2,567,538位用户和32,135个游戏。除评论文本外，数据还包括每个评论中用户的游戏时间。|[ICDM 2018](https://github.com/kang205/SASRec)|
 |[Douban](https://paddlerec.bj.bcebos.com/datasets/Douban/DMSC.csv)| -- |豆瓣电影是一个中文网站，允许互联网用户分享他们对电影的评论和观点。用户可以在电影上发表简短或长时间的评论并给他们打分。该数据集包含“豆瓣电影”网站中28部电影的200万条简短评论。|[Kaggle](https://www.kaggle.com/utmhikari/doubanmovieshortcomments)|
 |[TaFeng](https://paddlerec.bj.bcebos.com/datasets/tafeng/ta_feng_all_months_merged.csv)| -- |该数据集包含2000年11月至2001年2月中国杂货店的交易数据。|[Kaggle](https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset)|
 |[Retailrocket](https://paddlerec.bj.bcebos.com/datasets/Retailrocket/Retailrocket.zip)| -- |数据是从真实的电子商务网站中收集的。它是原始数据，即没有任何内容转换，但是，由于保密问题，所有值都被哈希化。|[Kaggle](https://www.kaggle.com/retailrocket/ecommerce-dataset)|
 |[Netflix](https://paddlerec.bj.bcebos.com/datasets/Netflix/Netflix.zip)| -- |这是Netflix竞赛中使用的官方数据集。|[Kaggle](https://www.kaggle.com/netflix-inc/netflix-prize-data)|
 |[FourSquare](https://paddlerec.bj.bcebos.com/datasets/FourSquare/FourSquare.zip)| -- |此数据集包含在纽约和东京进行的大约10个月收集的签到。每个签到都有其时间戳，GPS坐标及其语义相关联。|[Kaggle](https://www.kaggle.com/chetanism/foursquare-nyc-and-tokyo-checkin-dataset)|
 |[AmazonBook](https://paddlerec.bj.bcebos.com/datasets/AmazonBook/AmazonBook.tar.gz)| [mind](../models/recall/mind/mind_reader.py) |论文原作者处理过的AmazonBook数据集,[github](https://github.com/THUDM/ComiRec) |[《Controllable Multi-Interest Framework for Recommendation》](https://arxiv.org/abs/2005.09347)|
 |[Ali_Display_Ad_Click](https://paddlerec.bj.bcebos.com/datasets/dmr/dataset_full.zip)| [dmr](../models/rank/dmr/alimama_reader.py) |预处理过的Alimama数据集 |[Deep Match to Rank Model for Personalized Click-Through Rate Prediction](https://github.com/lvze92/DMR)|
 |[omniglot](https://paddlerec.bj.bcebos.com/datasets/omniglot/omniglot.tar)| [maml](../models/multitask/maml/omniglot_reader.py) |预处理过的omniglot数据集 |[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400.pdf)|
 |[LastFM](https://paddlerec.bj.bcebos.com/datasets/LastFM/lastfm-2k.zip)| -- |提供音乐推荐的数据集,对于数据集中的每个用户，包含他们最受欢迎的艺术家的列表以及播放次数 |[HetRec 2011](https://grouplens.org/datasets/hetrec-2011/)|
 |[Epinions](https://paddlerec.bj.bcebos.com/datasets/Epinions/epinions_data.tar.gz)| -- |Epinions数据集由一个普通消费者评论网站Epinions.com的who信任who在线社交网络构建 |[Epinions](https://snap.stanford.edu/data/soc-Epinions1.html)|
 |[Yelp](https://paddlerec.bj.bcebos.com/datasets/Epinions/soc-Epinions1.txt.gz)| -- |Yelp数据集是我们用于个人，教育和学术目的的业务，评论和用户数据的子集。以JSON文件形式提供，可用于在学习如何制作移动应用程序的同时，教给学生有关数据库，学习NLP或用于样例生产数据的信息。 |[Yelp](https://www.yelp.com/dataset)|
 |[book-crossing](https://paddlerec.bj.bcebos.com/datasets/book-crossing/BX-CSV-Dump.zip)| -- |Book-Crossings是由Cai-Nicolas Ziegler根据  bookcrossing.com 的数据编写的图书评分数据集。|[IIF](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)|
 |[Pinterest](https://paddlerec.bj.bcebos.com/datasets/Pinterest/pinterest-20.train.rating)| -- |Pinterest数据集包含超过100万张与Pinterest用户相关联的图像。|[Learning Image and User Features for Recommendation in Social Networks](https://openaccess.thecvf.com/content_iccv_2015/papers/Geng_Learning_Image_and_ICCV_2015_paper.pdf)|
 |[Gowalla](https://paddlerec.bj.bcebos.com/datasets/Gowalla/loc-gowalla_totalCheckins.txt.gz)| -- |Gowalla是一个基于位置的社交网站，用户可以通过登录来分享自己的位置 |[SNAP](https://cs.stanford.edu/people/jure/pubs/mobile-kdd11.pdf)|
 |[YOOCHOOSE](https://paddlerec.bj.bcebos.com/datasets/yoochose/yoochoose-data.zip)| -- |欧洲一家在线零售商用户的点击和购买行为. |[ACM](https://dl.acm.org/doi/10.1145/2792838.2798723)|
 |[Avazu](https://www.kaggle.com/c/avazu-ctr-prediction/data)| -- |按时间顺序排列的10天点击率数据. |[kaggle](https://www.kaggle.com/c/avazu-ctr-prediction/data)|
 |[Phishing_Websites](https://paddlerec.bj.bcebos.com/datasets/Phishing_Websites/train.arff)| -- |预测网络钓鱼网站有效的特征. |[Phishing_Websites](https://archive.ics.uci.edu/ml/datasets/phishing+websites)|
 |[Adult](https://paddlerec.bj.bcebos.com/datasets/Adult/adult.data)| -- |部分1994年人口普查数据库. |[Adult](https://archive.ics.uci.edu/ml/datasets/adult)|
 |[Taobao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)| -- |淘宝网用户行为数据. |[Ali](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)|
 |[JD](https://paddlerec.bj.bcebos.com/datasets/JD/jdata_tfrecord.zip)| -- |来自京东的用户行为数据，用于流媒体推荐. |[JD](https://datascience.jd.com/page/opendataset.html)|
 |[movielens](http://files.grouplens.org/datasets/movielens/ml-1m.zip)| -- |电影推荐数据集. |[movielens](https://grouplens.org/datasets/movielens/)|
 |[Imagenet](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar)| -- |ImageNet项目是一个大型视觉数据库，用于视觉目标识别研究任务，该项目已手动标注了 1400 多万张图像. |[Imagenet](https://www.image-net.org/)|
