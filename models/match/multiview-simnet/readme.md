# Multi-view Simnet for Personalized recommendation

## 简介

在个性化推荐场景中，推荐系统给用户提供的项目（Item）列表通常是通过个性化的匹配模型计算出来的。在现实世界中，一个用户可能有很多个视角的特征，比如用户Id，年龄，项目的点击历史等。一个项目，举例来说，新闻资讯，也会有多种视角的特征比如新闻标题，新闻类别等。Multi-view Simnet模型是可以融合用户以及推荐项目的多个视角的特征并进行个性化匹配学习的一体化模型。这类模型在很多工业化的场景中都会被使用到，比如百度的Feed产品中。

本项目的目标是提供一个在个性化匹配场景下利用Paddle搭建的模型。Multi-view Simnet模型包括多个编码器模块，每个编码器被用在不同的特征视角上。当前，项目中提供Bag-of-Embedding编码器，Temporal-Convolutional编码器，和Gated-Recurrent-Unit编码器。我们会逐渐加入稀疏特征场景下比较实用的编码器到这个项目中。模型的训练方法，当前采用的是Pairwise ranking模式进行训练，即针对一对具有关联的User-Item组合，并随机产出一个Item作为负例进行排序学习。

## 模型超参
```
optimizer:
  class: Adam                        # 优化器类型
  learning_rate: 0.0001              # 学习率
  strategy: async                    # 参数更新方式
query_encoder: "bow"                 # 用户特征编码器
title_encoder: "bow"                 # item特征编码器
query_encode_dim: 128                # 用户编码器产出的特征维度
title_encode_dim: 128                # item编码器产出的特征维度
sparse_feature_dim: 1000001          # 用户特征及item特征，所有特征总个数
embedding_dim: 128                   # 特征维度
hidden_size: 128                     # 隐藏层维度
margin: 0.1                          # max margin for hinge-loss
```

## 快速开始
PaddleRec内置了demo小数据，方便用户快速使用模型，训练命令如下：
```bash
python -m paddlerec.run -m paddlerec.models.match.multiview-simnet
```

执行预测前，需更改config.yaml中的配置，具体改动如下：
```
workspace: "~/code/paddlerec/models/match/multiview-simnet"     # 改为当前config.yaml所在的绝对路径

#mode: runner1     # train
mode: runner2     # infer

runner:
- name: runner2
  class: single_infer
  init_model_path: "increment/2"       # 改为需要预测的模型路径

phase:
- name: phase1
  model: "{workspace}/model.py"
  dataset_name: dataset_infer          # 改成预测dataset
  thread_num: 1                        # dataset线程数
```
改完之后，执行预测命令：
```
python -m paddlerec.run -m ./config.yaml
```

## 提测说明
当前，Multi-view Simnet模型采用的数据集是机器随机构造的，因此提测仅需按上述步骤在demo数据集上跑通即可。
