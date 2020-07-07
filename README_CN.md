(简体中文|[English](./README.md))

<p align="center">
<img align="center" src="doc/imgs/logo.png">
<p>
<p align="center">
<img align="center" src="doc/imgs/structure.png">
<p>
<p align="center">
<img align="center" src="doc/imgs/overview.png">
<p>


<h2 align="center">什么是推荐系统?</h2>
<p align="center">
<img align="center" src="doc/imgs/rec-overview.png">
<p>

- 推荐系统是在互联网信息爆炸式增长的时代背景下，帮助用户高效获得感兴趣信息的关键；

- 推荐系统也是帮助产品最大限度吸引用户、留存用户、增加用户粘性、提高用户转化率的银弹。

- 有无数优秀的产品依靠用户可感知的推荐系统建立了良好的口碑，也有无数的公司依靠直击用户痛点的推荐系统在行业中占领了一席之地。

  > 可以说，谁能掌握和利用好推荐系统，谁就能在信息分发的激烈竞争中抢得先机。
  > 但与此同时，有着许多问题困扰着推荐系统的开发者，比如：庞大的数据量，复杂的模型结构，低效的分布式训练环境，波动的在离线一致性，苛刻的上线部署要求，以上种种，不胜枚举。

<h2 align="center">什么是PaddleRec?</h2>


- 源于飞桨生态的搜索推荐模型 **一站式开箱即用工具** 
- 适合初学者，开发者，研究者的推荐系统全流程解决方案
- 包含内容理解、匹配、召回、排序、 多任务、重排序等多个任务的完整推荐搜索算法库


    |   方向   |                                   模型                                    | 单机CPU | 单机GPU | 分布式CPU | 分布式GPU | 论文                                                                                                                                                                                                        |
    | :------: | :-----------------------------------------------------------------------: | :-----: | :-----: | :-------: | :-------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | 内容理解 | [Text-Classifcation](models/contentunderstanding/classification/model.py) |    ✓    |    ✓    |     ✓     |     x     | [EMNLP 2014][Convolutional neural networks for sentence classication](https://www.aclweb.org/anthology/D14-1181.pdf)                                                                                        |
    | 内容理解 |         [TagSpace](models/contentunderstanding/tagspace/model.py)         |    ✓    |    ✓    |     ✓     |     x     | [EMNLP 2014][TagSpace: Semantic Embeddings from Hashtags](https://www.aclweb.org/anthology/D14-1194.pdf)                                                                                                    |
    |   匹配   |                    [DSSM](models/match/dssm/model.py)                     |    ✓    |    ✓    |     ✓     |     x     | [CIKM 2013][Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)             |
    |   匹配   |        [MultiView-Simnet](models/match/multiview-simnet/model.py)         |    ✓    |    ✓    |     ✓     |     x     | [WWW 2015][A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf)             |
    |   召回   |                   [TDM](models/treebased/tdm/model.py)                    |    ✓    | >=1.8.0 |     ✓     |  >=1.8.0  | [KDD 2018][Learning Tree-based Deep Model for Recommender Systems](https://arxiv.org/pdf/1801.02294.pdf)                                                                                                    |
    |   召回   |                [fasttext](models/recall/fasttext/model.py)                |    ✓    |    ✓    |     x     |     x     | [EACL 2017][Bag of Tricks for Efficient Text Classification](https://www.aclweb.org/anthology/E17-2068.pdf)                                                                                                 |
    |   召回   |                [Word2Vec](models/recall/word2vec/model.py)                |    ✓    |    ✓    |     ✓     |     x     | [NIPS 2013][Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) |
    |   召回   |                     [SSR](models/recall/ssr/model.py)                     |    ✓    |    ✓    |     ✓     |     ✓     | [SIGIR 2016][Multi-Rate Deep Learning for Temporal Recommendation](http://sonyis.me/paperpdf/spr209-song_sigir16.pdf)                                                                                       |
    |   召回   |                 [Gru4Rec](models/recall/gru4rec/model.py)                 |    ✓    |    ✓    |     ✓     |     ✓     | [2015][Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)                                                                                                      |
    |   召回   |             [Youtube_dnn](models/recall/youtube_dnn/model.py)             |    ✓    |    ✓    |     ✓     |     ✓     | [RecSys 2016][Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)                                               |
    |   召回   |                     [NCF](models/recall/ncf/model.py)                     |    ✓    |    ✓    |     ✓     |     ✓     | [WWW 2017][Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)                                                                                                                            |
    |   召回   |                     [GNN](models/recall/gnn/model.py)                     |    ✓    |    ✓    |     ✓     |     ✓     | [AAAI 2019][Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855)                                                                                                      |
    |   排序   |      [Logistic Regression](models/rank/logistic_regression/model.py)      |    ✓    |    x    |     ✓     |     x     | /                                                                                                                                                                                                           |
    |   排序   |                      [Dnn](models/rank/dnn/model.py)                      |    ✓    |    ✓    |     ✓     |     ✓     | /                                                                                                                                                                                                           |
    |   排序   |                       [FM](models/rank/fm/model.py)                       |    ✓    |    x    |     ✓     |     x     | [IEEE Data Mining 2010][Factorization machines](https://analyticsconsultores.com.mx/wp-content/uploads/2019/03/Factorization-Machines-Steffen-Rendle-Osaka-University-2010.pdf)                             |
    |   排序   |                      [FFM](models/rank/ffm/model.py)                      |    ✓    |    x    |     ✓     |     x     | [RECSYS 2016][Field-aware Factorization Machines for CTR Prediction](https://dl.acm.org/doi/pdf/10.1145/2959100.2959134)                                                                                    |
    |   排序   |                      [FNN](models/rank/fnn/model.py)                      |    ✓    |    x    |     ✓     |     x     | [ECIR 2016][Deep Learning over Multi-field Categorical Data](https://arxiv.org/pdf/1601.02376.pdf)                                                                                                          |
    |   排序   |            [Deep Crossing](models/rank/deep_crossing/model.py)            |    ✓    |    x    |     ✓     |     x     | [ACM 2016][Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)                                                   |
    |   排序   |                      [Pnn](models/rank/pnn/model.py)                      |    ✓    |    x    |     ✓     |     x     | [ICDM 2016][Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                                                               |
    |   排序   |                      [DCN](models/rank/dcn/model.py)                      |    ✓    |    x    |     ✓     |     x     | [KDD 2017][Deep & Cross Network for Ad Click Predictions](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754)                                                                                               |
    |   排序   |                      [NFM](models/rank/nfm/model.py)                      |    ✓    |    x    |     ✓     |     x     | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/doi/pdf/10.1145/3077136.3080777)                                                                             |
    |   排序   |                      [AFM](models/rank/afm/model.py)                      |    ✓    |    x    |     ✓     |     x     | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)                                                  |
    |   排序   |                   [DeepFM](models/rank/deepfm/model.py)                   |    ✓    |    x    |     ✓     |     x     | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)                                                                                 |
    |   排序   |                  [xDeepFM](models/rank/xdeepfm/model.py)                  |    ✓    |    x    |     ✓     |     x     | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/3219819.3220023)                                                       |
    |   排序   |                      [DIN](models/rank/din/model.py)                      |    ✓    |    x    |     ✓     |     x     | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://dl.acm.org/doi/pdf/10.1145/3219819.3219823)                                                                                     |
    |   排序   |                      [DIEN](models/rank/dien/model.py)                      |    ✓    |    x    |     ✓     |     x     | [AAAI 2019][Deep Interest Evolution Network for Click-Through Rate Prediction](https://www.aaai.org/ojs/index.php/AAAI/article/view/4545/4423)                                                                                     |
    |   排序   |                      [AutoInt](models/rank/AutoInt/model.py)                      |    ✓    |    x    |     ✓     |     x     | [CIKM 2019][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)                                                                                     |
    |   排序   |                [Wide&Deep](models/rank/wide_deep/model.py)                |    ✓    |    x    |     ✓     |     x     | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454)                                                                                               |
    |   排序   |                    [FGCNN](models/rank/fgcnn/model.py)                    |    ✓    |    ✓    |     ✓     |     ✓     | [WWW 2019][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1904.04447.pdf)                                                                      |
    |   排序   |                  [Fibinet](models/rank/fibinet/model.py)                  |    ✓    |    ✓    |     ✓     |     ✓     | [RecSys19][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction]( https://arxiv.org/pdf/1905.09433.pdf)                                                 |
    |  多任务  |                  [ESMM](models/multitask/esmm/model.py)                   |    ✓    |    ✓    |     ✓     |     ✓     | [SIGIR 2018][Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931)                                                              |
    |  多任务  |                  [MMOE](models/multitask/mmoe/model.py)                   |    ✓    |    ✓    |     ✓     |     ✓     | [KDD 2018][Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007)                                                       |
    |  多任务  |           [ShareBottom](models/multitask/share-bottom/model.py)           |    ✓    |    ✓    |     ✓     |     ✓     | [1998][Multitask learning](http://reports-archive.adm.cs.cmu.edu/anon/1997/CMU-CS-97-203.pdf)                                                                                                               |
    |  重排序  |                [Listwise](models/rerank/listwise/model.py)                |    ✓    |    ✓    |     ✓     |     x     | [2019][Sequential Evaluation and Generation Framework for Combinatorial Recommender System](https://arxiv.org/pdf/1902.00245.pdf)                                                                           |





<h2 align="center">快速安装</h2>

### 环境要求
* Python 2.7/ 3.5 / 3.6 / 3.7
* PaddlePaddle  >= 1.7.2
* 操作系统: Windows/Mac/Linux

  > Windows下PaddleRec目前仅支持单机训练，分布式训练建议使用Linux环境
  
### 安装命令

- 安装方法一 **PIP源直接安装**
  ```bash
  python -m pip install paddle-rec
  ```
    > 该方法会默认下载安装`paddlepaddle v1.7.2 cpu版本`，若提示`PaddlePaddle`无法安装，则依照下述方法首先安装`PaddlePaddle`，再安装`PaddleRec`：
    > - 可以在[该地址](https://pypi.org/project/paddlepaddle/1.7.2/#files)，下载PaddlePaddle后手动安装whl包
    > - 可以先pip安装`PaddlePaddle`，`python -m pip install paddlepaddle==1.7.2 -i https://mirror.baidu.com/pypi/simple`
    > - 其他安装问题可以在[Paddle Issue](https://github.com/PaddlePaddle/Paddle/issues)或[PaddleRec Issue](https://github.com/PaddlePaddle/PaddleRec/issues)提出，会有工程师及时解答

- 安装方法二 **源码编译安装**
  
  - 安装飞桨  **注：需要用户安装版本 == 1.7.2 的飞桨**

    ```shell
    python -m pip install paddlepaddle==1.7.2 -i https://mirror.baidu.com/pypi/simple
    ```

  - 源码安装PaddleRec

    ```
    git clone https://github.com/PaddlePaddle/PaddleRec/
    cd PaddleRec
    python setup.py install
    ```

- PaddleRec-GPU安装方法

  在使用方法一或方法二完成PaddleRec安装后，需再手动安装`paddlepaddle-gpu`，并根据自身环境(Cuda/Cudnn)选择合适的版本，安装教程请查阅[飞桨-开始使用](https://www.paddlepaddle.org.cn/install/quick)


<h2 align="center">一键启动</h2>

我们以排序模型中的`dnn`模型为例介绍PaddleRec的一键启动。训练数据来源为[Criteo数据集](https://www.kaggle.com/c/criteo-display-ad-challenge/)，我们从中截取了100条数据：

```bash
# 使用CPU进行单机训练
python -m paddlerec.run -m paddlerec.models.rank.dnn  
```


<h2 align="center">帮助文档</h2>

### 项目背景
* [推荐系统介绍](doc/rec_background.md)
* [分布式深度学习介绍](doc/ps_background.md)

### 快速开始
* [十分钟上手PaddleRec](https://aistudio.baidu.com/aistudio/projectdetail/559336)

### 入门教程
* [数据准备](doc/slot_reader.md)
* [模型调参](doc/model.md)
* [启动训练](doc/train.md)
* [启动预测](doc/predict.md)
* [快速部署](doc/serving.md)


### 进阶教程
* [自定义Reader](doc/custom_reader.md)
* [自定义模型](doc/model_develop.md)
* [自定义流程](doc/trainer_develop.md)
* [yaml配置说明](doc/yaml.md)
* [PaddleRec设计文档](doc/design.md)

### Benchmark
* [Benchmark](doc/benchmark.md)

### FAQ
* [常见问题FAQ](doc/faq.md)


<h2 align="center">社区</h2>

<p align="center">
    <br>
    <img alt="Release" src="https://img.shields.io/badge/Release-0.1.0-yellowgreen">
    <img alt="License" src="https://img.shields.io/github/license/PaddlePaddle/PaddleRec">
    <img alt="Slack" src="https://img.shields.io/badge/Join-Slack-green">
    <br>
<p>

### 版本历史
- 2020.06.17 - PaddleRec v0.1.0
- 2020.06.03 - PaddleRec v0.0.2
- 2020.05.14 - PaddleRec v0.0.1
  
### 许可证书
本项目的发布受[Apache 2.0 license](LICENSE)许可认证。

### 联系我们

如有意见、建议及使用中的BUG，欢迎在[GitHub Issue](https://github.com/PaddlePaddle/PaddleRec/issues)提交

亦可通过以下方式与我们沟通交流：

- QQ群号码：`861717190`
- 微信小助手微信号：`paddlerec2020`

<p align="center"><img width="200" height="200" margin="500" src="./doc/imgs/QQ_group.png"/>&#8194;&#8194;&#8194;&#8194;&#8194<img width="200" height="200"  src="doc/imgs/weixin_supporter.png"/></p>
<p align="center">PaddleRec交流QQ群&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;PaddleRec微信小助手</p>
