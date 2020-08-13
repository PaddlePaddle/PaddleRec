# 匹配模型库

## 简介
我们提供了常见的匹配任务中使用的模型算法的PaddleRec实现, 单机训练&预测效果指标以及分布式训练&预测性能指标等。实现的模型包括 [DSSM](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/match/dssm)、[MultiView-Simnet](http://gitlab.baidu.com/tangwei12/paddlerec/tree/develop/models/match/multiview-simnet)。

模型算法库在持续添加中，欢迎关注。

## 目录
* [整体介绍](#整体介绍)
    * [匹配模型列表](#匹配模型列表)
* [使用教程](#使用教程)
    * [训练&预测](#训练&预测)

## 整体介绍
### 匹配模型列表

|       模型        |       简介        |       论文        |
| :------------------: | :--------------------: | :---------: |
| DSSM | Deep Structured Semantic Models | [CIKM 2013][Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf) |
| MultiView-Simnet | Multi-view Simnet for Personalized recommendation | [WWW 2015][A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf) |

下面是每个模型的简介（注：图片引用自链接中的论文）

[DSSM](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf):
<p align="center">
<img align="center" src="../../doc/imgs/dssm.png">
<p>

[MultiView-Simnet](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf):
<p align="center">
<img align="center" src="../../doc/imgs/multiview-simnet.png">
<p>

## 使用教程(快速开始)
### 训练
```shell
git clone https://github.com/PaddlePaddle/PaddleRec.git paddle-rec
cd paddle-rec

python -m paddlerec.run -m models/match/dssm/config.yaml # dssm
python -m paddlerec.run -m models/match/multiview-simnet/config.yaml # multiview-simnet
```

### 预测
```shell
# 修改对应模型的config.yaml, workspace配置为当前目录的绝对路径
# 修改对应模型的config.yaml，mode配置infer_runner
# 示例: mode: train_runner -> mode: infer_runner
# infer_runner中 class配置为 class: infer
# 修改phase阶段为infer的配置，参照config注释

# 修改完config.yaml后 执行:
python -m paddlerec.run -m ./config.yaml # 以dssm为例
```
