<p align="center">
<img align="center" src="doc/imgs/logo.png">
<p>
<p align="center">
<img align="center" src="doc/imgs/structure.png">
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
- 包含语义理解、召回、粗排、精排、多任务学习、融合等多个任务的完整推荐搜索算法库


    |   方向   |                                   模型                                    | 单机CPU训练 | 单机GPU训练 | 分布式CPU训练 |
    | :------: | :-----------------------------------------------------------------------: | :---------: | :---------: | :-----------: |
    | 内容理解 | [Text-Classifcation](models/contentunderstanding/classification/model.py) |      ✓      |      x      |       ✓       |
    | 内容理解 |         [TagSpace](models/contentunderstanding/tagspace/model.py)         |      ✓      |      x      |       ✓       |
    |   召回   |                    [DSSM](models/match/dssm/model.py)                     |      ✓      |      x      |       ✓       |
    |   召回   |        [MultiView-Simnet](models/match/multiview-simnet/model.py)         |      ✓      |      x      |       ✓       |
    |   召回   |                   [TDM](models/treebased/tdm/model.py)                    |      ✓      |      x      |       ✓       |
    |   召回   |                [Word2Vec](models/recall/word2vec/model.py)                |      ✓      |      x      |       ✓       |
    |   召回   |                     [SSR](models/recall/ssr/model.py)                     |      ✓      |      ✓      |       ✓       |
    |   召回   |                 [Gru4Rec](models/recall/gru4rec/model.py)                 |      ✓      |      ✓      |       ✓       |
    |   召回   |             [Youtube_dnn](models/recall/youtube_dnn/model.py)             |      ✓      |      ✓      |       ✓       |
    |   召回   |                     [NCF](models/recall/ncf/model.py)                     |      ✓      |      ✓      |       ✓       |
    |   排序   |                      [Dnn](models/rank/dnn/model.py)                      |      ✓      |      x      |       ✓       |
    |   排序   |                   [DeepFM](models/rank/deepfm/model.py)                   |      ✓      |      x      |       ✓       |
    |   排序   |                  [xDeepFM](models/rank/xdeepfm/model.py)                  |      ✓      |      x      |       ✓       |
    |   排序   |                      [DIN](models/rank/din/model.py)                      |      ✓      |      x      |       ✓       |
    |   排序   |                [Wide&Deep](models/rank/wide_deep/model.py)                |      ✓      |      x      |       ✓       |
    |  多任务  |                  [ESMM](models/multitask/esmm/model.py)                   |      ✓      |      ✓      |       ✓       |
    |  多任务  |                  [MMOE](models/multitask/mmoe/model.py)                   |      ✓      |      ✓      |       ✓       |
    |  多任务  |           [ShareBottom](models/multitask/share-bottom/model.py)           |      ✓      |      ✓      |       ✓       |
    |  重排序  |                [Listwise](models/rerank/listwise/model.py)                |      ✓      |      x      |       ✓       |





<h2 align="center">快速安装</h2>

### 环境要求
* Python 2.7/ 3.5 / 3.6 / 3.7
* PaddlePaddle  >= 1.7.2
* 操作系统: Windows/Mac/Linux

  > Windows下目前仅提供单机训练，建议使用Linux
  
### 安装命令

- 安装方法一<PIP源直接安装>：
  ```bash
  python -m pip install paddle-rec
  ```

- 安装方法二

  源码编译安装
  1. 安装飞桨  **注：需要用户安装版本 >1.7.2 的飞桨**

    ```shell
    python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
    ```

  2. 源码安装PaddleRec

    ```
    git clone https://github.com/PaddlePaddle/PaddleRec/
    cd PaddleRec
    python setup.py install
    ```


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
* [十分钟上手PaddleRec](doc/quick_start.md)

### 入门教程
* [数据准备](doc/slot_reader.md)
* [模型调参](doc/model.md)
* [启动训练](doc/train.md)
* [启动预测](doc/predict.md)
* [快速部署](doc/serving.md)


### 进阶教程
* [自定义Reader](doc/custom_reader.md)
* [自定义模型](doc/model_development.md)
* [自定义流程](doc/model_development.md)
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
    <img alt="License" src="https://img.shields.io/github/license/PaddlePaddle/Serving">
    <img alt="Slack" src="https://img.shields.io/badge/Join-Slack-green">
    <br>
<p>

### 版本历史
- 2020.5.14 - PaddleRec v0.1
  
### 许可证书
本项目的发布受[Apache 2.0 license](LICENSE)许可认证。

### 联系我们

如有意见、建议及使用中的BUG，欢迎在[GitHub Issue](https://github.com/PaddlePaddle/PaddleRec/issues)提交

亦可通过以下方式与我们沟通交流：

<p align="center"><img width="200" height="200"  src="doc/imgs/wechet.png"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="200" height="200" margin="500" src="./doc/imgs/QQ_group.png"/>&#8194;&#8194;&#8194;&#8194;&#8194<img width="200" height="200"  src="doc/imgs/weixin_supporter.png"/></p>
<p align="center">  &#8194;&#8194;&#8194;&#8194;&#8194;&#8194;微信公众号&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;PaddleRec交流QQ群&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;PaddleRec微信小助手</p>