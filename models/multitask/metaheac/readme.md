# MetaHeac

以下是本例的简要目录结构及说明：

```
├── data #样例数据
    ├── train #训练数据
        ├── train_stage1.pkl
    ├── test #测试数据
        ├── test_stage1.pkl
        ├── test_stage2.pkl
├── net.py # 核心模型组网
├── config.yaml # sample数据配置
├── config_big.yaml # 全量数据配置
├── dygraph_model.py # 构建动态图
├── reader_train.py # 训练数据读取程序
├── reader_test.py # infer数据读取程序
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
在推荐系统和广告平台上，营销人员总是希望通过视频或者社交等媒体渠道向潜在用户推广商品、内容或者广告。扩充候选集技术（Look-alike建模）是一种很有效的解决方案，但look-alike建模通常面临两个挑战：（1）一家公司每天可以开展数百场营销活动，以推广完全不同类别的各种内容。（2）某项活动的种子集只能覆盖有限的用户，因此一个基于有限种子用户的定制化模型往往会产生严重的过拟合。为了解决以上的挑战，论文《Learning to Expand Audience via Meta Hybrid Experts and Critics for Recommendation and Advertising》提出了一种新的两阶段框架Meta Hybrid Experts and Critics (MetaHeac)，采用元学习的方法训练一个泛化初始化模型，从而能够快速适应新类别内容推广任务。

## 数据准备
使用Tencent Look-alike Dataset,该数据集包含几百个种子人群、海量候选人群对应的用户特征，以及种子人群对应的广告特征。出于业务数据安全保证的考虑，所有数据均为脱敏处理后的数据。本次复现使用处理过的数据集，直接下载[propocessed data](https://drive.google.com/file/d/11gXgf_yFLnbazjx24ZNb_Ry41MI5Ud1g/view?usp=sharing),mataheac/data/目录下存放了从全量数据集获取的少量数据集，用于对齐模型。

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos

## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在MetaHeac模型目录的快速执行命令如下：
```bash
# 进入模型目录
%cd PaddleRec/models/multitask/metaheac/
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
# 进入数据集目录,并运行run.sh下载数据集
%cd PaddleRec/datasets/Lookalike/
!sed -i 's/\r$//' run.sh
!sh run.sh
# 退回根目录
%cd ../../../
# 进入模型目录
%cd PaddleRec/models/multitask/metaheac/
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
