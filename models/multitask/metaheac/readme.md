# MetaHeac
论文复现赛-MetaHeac
# MetaHeac模型

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── test
    ├── train        
├── net.py # 模型组网
├── config.yaml # 测试数据配置文件
├── config_big.yaml # 全量数据配置文件
├── dygraph_model.py # 动态图训练
├── reader_train.py # 读取训练数据集
├── reader_test.py # 读取infer数据集
├── readme.md #文档
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [模型组网](#模型组网)
- [效果复现](#效果复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)

## 模型简介
Package Recommendation with Intra- and Inter-Package Attention Networks 《利用“包内”和“包间”注意力网络的包推荐》。  
随着移动互联网中在线社交网络的蓬勃发展，我们提出了一个新颖的社交推荐场景，名为Package Recommendation。在这种场景中，用户不再被推荐单个项目或项目列表，而是被推荐异构且类型多样对象的组合（称为包，例如，包括新闻、媒体和观看新闻的朋友）。与传统推荐不同，在包推荐中，包中的对象被明确显示给用户，用户会对显式展示的对象表现出极大的兴趣，反过来这些对象可能对用户行为产生重大影响，并显著改变传统的推荐模式。

## 数据准备
使用Tencent Look-alike Dataset,该数据集包含几百个种子人群、海量候选人群对应的用户特征，以及种子人群对应的广告特征。出于业务数据安全保证的考虑，所有数据均为脱敏处理后的数据。本次复现使用处理过的数据集，直接下载即可 propocessed data[https://drive.google.com/file/d/11gXgf_yFLnbazjx24ZNb_Ry41MI5Ud1g/view?usp=sharing]
data/processed_data目录下存放了从全量数据集获取的少量数据集，做对齐模型使用。
若需要使用全量数据可以参考下方效果复现部分。
## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在MetaHeac模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# %cd PaddleRec-master/models/multitask/metaheac/

# 动态图训练
# step1： train
!python -W ignore -u ../../../tools/trainer.py -m config.yaml

# 动态图预测
# step2： infer 此时test数据集为hot
!python -W ignore -u ../../../tools/infer_meta.py -m config.yaml
# step3：修改config文件中test文件的路径为cold
# !python -W ignore -u ../../../tools/infer_meta.py -m config.yaml
```
## 模型组网
模型整体结构如下： 


## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。  
在全量数据下模型的指标如下(train.py文件内 paddle.seed = 2021下效果)：  

| 模型    | auc    | batch_size | epoch_num| Time of each epoch |
|:------|:-------| :------ | :------| :------ |
| MetaHeac | 0.7112 | 1024 | 1 | 3个小时左右 |

```bash
# 进入模型目录
# %cd PaddleRec-master/models/multitask/metaheac/

# 动态图训练
# step1： train
!python -W ignore -u ../../../tools/trainer.py -m config_big.yaml

# 动态图预测
# step2： infer 此时test数据集为hot
!python -W ignore -u ../../../tools/infer_meta.py -m config_big.yaml
# step3：修改config文件中test文件的路径为cold
# !python -W ignore -u ../../../tools/infer_meta.py -m config.yaml
```

## 进阶使用
  
## FAQ