# 匹配模型库

## 简介
我们提供了常见的匹配任务中使用的模型算法的PaddleRec实现, 包括动态图和静态图的单机训练&预测效果指标。实现的模型包括 [DSSM](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/match/dssm)、[MultiView-Simnet](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/match/multiview-simnet)、[match-pyramid](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/match/match-pyramid)。  
模型算法库在持续添加中，欢迎关注。

## 目录
* [整体介绍](#整体介绍)
    * [模型列表](#模型列表)
* [使用教程](#使用教程)
    * [快速开始](#快速开始)
    * [模型效果](#模型效果)
    * [效果复现](#效果复现)

## 整体介绍
### 模型列表

|       模型        |       简介        |       论文        |
| :------------------: | :--------------------: | :---------: |
| DSSM | Deep Structured Semantic Models | [CIKM 2013][Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf) |
| MultiView-Simnet | Multi-view Simnet for Personalized recommendation | [WWW 2015][A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf) |
| match-pyramid | Text Matching as Image Recognition | [arXiv W2016][Text Matching as Image Recognition](https://arxiv.org/pdf/1602.06359.pdf) |


下面是每个模型的简介（注：图片引用自链接中的论文）

[DSSM](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf):
<p align="center">
<img align="center" src="../../doc/imgs/dssm.png">
<p>

[MultiView-Simnet](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf):
<p align="center">
<img align="center" src="../../doc/imgs/multiview-simnet.png">
<p>

[match-pyramid](https://arxiv.org/pdf/1602.06359.pdf):
<p align="center">
<img align="center" src="../../doc/imgs/match-pyramid.png">
<p>

## 使用教程

### 快速开始
```bash
# 进入模型目录
cd models/match/xxx # xxx为任意的match下的模型目录
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 动态图预测
python -u ../../../tools/infer.py -m config.yaml 

# 静态图训练
python -u ../../../tools/static_trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 静态图预测
python -u ../../../tools/static_infer.py -m config.yaml 
```

### 模型效果

|       数据集        |       模型       |      正序率          |       map       |  
| :------------------: | :--------------------: | :---------: |:---------: |
|       BQ       |       DSSM       |       0.79        |       --          | 
|       Letor07        |       match-pyramid       |       --        |      0.39          | 
|       BQ        |       multiview-simnet       |       0.82        |       --          |

### 效果复现
您需要进入PaddleRec/datasets目录下的对应数据集中运行脚本获取全量数据集，然后在模型目录下使用全量数据的参数运行。  
每个模型下的readme中都有详细的效果复现的教程，您可以进入模型的目录中详细查看。  
