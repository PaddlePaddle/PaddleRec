<p align="center">
<img align="center" src="doc/imgs/logo.png">
<p>

<p align="center">
    <br>
    <a href="https://travis-ci.com/PaddlePaddle/Serving">
        <img alt="Build Status" src="https://img.shields.io/travis/com/PaddlePaddle/Serving/develop">
    </a>
    <img alt="Release" src="https://img.shields.io/badge/Release-0.0.3-yellowgreen">
    <img alt="Issues" src="https://img.shields.io/github/issues/PaddlePaddle/Serving">
    <img alt="License" src="https://img.shields.io/github/license/PaddlePaddle/Serving">
    <img alt="Slack" src="https://img.shields.io/badge/Join-Slack-green">
    <br>
<p>


<h2 align="center">什么是PaddleRec</h2>

<p align="center">
<img align="center" src="doc/imgs/structure.png">
<p>

- PaddleRec是源于飞桨生态的搜索推荐模型一站式开箱即用工具，无论您是初学者，开发者，研究者均可便捷的使用PaddleRec完成调研，训练到预测部署的全流程工作。
- PaddleRec提供了搜索推荐任务中语义理解、召回、粗排、精排、多任务学习的全流程解决方案，包含的算法模型均在百度各个业务的实际场景中得到了验证。
- PaddleRec将各个模型及其训练预测流程规范化整理，进行易用性封装，用户只需自定义yaml文件即可快速上手使用。
- PaddleRec以飞桨深度学习框架为核心，融合了大规模分布式训练框架Fleet，以及一键式推理部署框架PaddleServing，支持推荐搜索算法的工业化应用。


<h2 align="center">PadlleRec概览</h2>

<p align="center">
<img align="center" src="doc/imgs/overview.png">
<p>


<h2 align="center">便捷安装</h2>

### 环境要求
* Python >= 2.7
* PaddlePaddle >= 1.7.2
* 操作系统: Windows/Mac/Linux
  
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

  2. 源码安装Fleet-Rec

    ```
    git clone https://github.com/PaddlePaddle/PaddleRec/
    cd PaddleRec
    python setup.py install
    ```


<h2 align="center">快速启动</h2>


目前框架内置了多个模型，简单的命令即可使用内置模型开始单机训练和本地1*1模拟训练，我们以`ctr-dnn`为例介绍PaddleRec的简单使用。

### 一行命令启动训练

<h3 align="center">单机训练</h3>

```bash
# 使用CPU进行单机训练
python -m fleetrec.run -m fleetrec.models.rank.dnn -d cpu -e single 

# 使用GPU进行单机训练
python -m fleetrec.run -m fleetrec.models.rank.dnn -d gpu -e single
```

<h3 align="center">本地模拟分布式训练</h3>

```bash
# 使用CPU资源进行本地模拟分布式训练
python -m fleetrec.run -m fleetrec.models.rank.dnn -d cpu -e local_cluster
```

<h3 align="center">集群分布式训练</h3>

```bash
# 配置好 mpi/k8s/paddlecloud集群环境后
python -m fleetrec.run -m fleetrec.models.rank.dnn -d cpu -e cluster
```

<h2 align="center">支持模型列表</h2>

> 部分表格占位待改（大规模稀疏）

|   方向   |                                      模型                                      | 单机CPU训练 | 单机GPU训练 | 分布式CPU训练 | 大规模稀疏 | 分布式GPU训练 | 自定义数据集 |
| :------: | :----------------------------------------------------------------------------: | :---------: | :---------: | :-----------: | :--------: | :-----------: | :----------: |
| 内容理解 | [Text-Classifcation](models/contentunderstanding/text_classification/model.py) |      ✓      |      x      |       ✓       |     x      |       ✓       |      ✓       |
| 内容理解 |           [TagSpace](models/contentunderstanding/tagspace/model.py)            |      ✓      |      x      |       ✓       |     x      |       ✓       |      ✓       |
|   召回   |                  [Word2Vec](models/recall/word2vec/model.py)                   |      ✓      |      x      |       ✓       |     x      |       ✓       |      ✓       |
|   召回   |                       [TDM](models/recall/tdm/model.py)                        |      ✓      |      x      |       ✓       |     x      |       ✓       |      ✓       |
|   召回   |                       [SSR](models/recall/ssr/model.py)                        |      ✓      |      ✓      |       ✓       |     x      |       ✓       |      ✓       |
|   召回   |                   [Gru4Rec](models/recall/gru4rec/model.py)                    |      ✓      |      ✓      |       ✓       |     x      |       ✓       |      ✓       |
|   排序   |                      [CTR-Dnn](models/rank/dnn/model.py)                       |      ✓      |      x      |       ✓       |     x      |       ✓       |      ✓       |
|   排序   |                     [DeepFm](models/rank/deepfm/model.py)                      |      ✓      |      x      |       ✓       |     x      |       ✓       |      ✓       |
|   排序   |                    [xDeepFm](models/rank/xdeepfm/model.py)                     |      ✓      |      x      |       ✓       |     x      |       ✓       |      ✓       |
|   排序   |                        [DIN](models/rank/din/model.py)                         |      ✓      |      x      |       ✓       |     x      |       ✓       |      ✓       |
|   排序   |                  [Wide&Deep](models/rank/wide_deep/model.py)                   |      ✓      |      x      |       ✓       |     x      |       ✓       |      ✓       |
|  多任务  |                     [ESMM](models/multitask/essm/model.py)                     |      ✓      |      ✓      |       ✓       |     x      |       ✓       |      ✓       |
|  多任务  |                     [MMOE](models/multitask/mmoe/model.py)                     |      ✓      |      ✓      |       ✓       |     x      |       ✓       |      ✓       |
|   排序   |             [ShareBottom](models/multitask/share-bottom/model.py)              |      ✓      |      ✓      |       ✓       |     x      |       ✓       |      ✓       |
|   匹配   |                       [DSSM](models/match/dssm/model.py)                       |      ✓      |      x      |       ✓       |     x      |       ✓       |      ✓       |
|   匹配   |                [Simnet](models/match/multiview-simnet/model.py)                |      ✓      |      x      |       ✓       |     x      |       ✓       |      ✓       |





<h2 align="center">文档</h2>

### 新手教程
* [环境要求](#环境要求)
* [安装命令](#安装命令)
* [快速开始](#一行命令启动训练)
  
### 进阶教程
* [自定义数据集及Reader](doc/custom_dataset_reader.md)
* [模型调参](doc/optimization_model.md)
* [单机训练](doc/local_train.md)
* [分布式训练](doc/distributed_train.md)
* [离线预测](doc/predict.md)

### 关于PaddleRec性能
* [Benchamrk](doc/benchmark.md)

### FAQ
* [常见问题FAQ](doc/faq.md)

### 设计文档
* [PaddleRec设计文档](doc/design.md)


<h2 align="center">社区</h2>

### 贡献代码
* [优化PaddleRec框架](doc/contribute.md)
* [新增模型到PaddleRec](doc/contribute.md)

### 反馈
如有意见、建议及使用中的BUG，欢迎在`GitHub Issue`提交

### 版本历史
* [版本更新](#版本更新)
  
### 许可证书
本项目的发布受[Apache 2.0 license](LICENSE)许可认证。
  