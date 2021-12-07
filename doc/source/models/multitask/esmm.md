# esmm (Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate)

代码请参考：[ESMM](https://github.com/PaddlePaddle/PaddleRec/blob/master/models/multitask/esmm)  
如果我们的代码对您有用，还请点个star啊~  

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
不同于CTR预估问题，CVR预估面临两个关键问题：

1. **Sample Selection Bias (SSB)** 转化是在点击之后才“有可能”发生的动作，传统CVR模型通常以点击数据为训练集，其中点击未转化为负例，点击并转化为正例。但是训练好的模型实际使用时，则是对整个空间的样本进行预估，而非只对点击样本进行预估。即是说，训练数据与实际要预测的数据来自不同分布，这个偏差对模型的泛化能力构成了很大挑战。
2. **Data Sparsity (DS)** 作为CVR训练数据的点击样本远小于CTR预估训练使用的曝光样本。

ESMM是发表在 SIGIR’2018 的论文[《Entire Space Multi-Task Model: An Eﬀective Approach for Estimating Post-Click Conversion Rate》](  https://arxiv.org/abs/1804.07931  )文章基于 Multi-Task Learning 的思路，提出一种新的CVR预估模型——ESMM，有效解决了真实场景中CVR预估面临的数据稀疏以及样本选择偏差这两个关键问题

## 数据准备
我们在开源数据集[Ali-CCP：Alibaba Click and Conversion Prediction](  https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408  )上验证模型效果。在模型目录的data目录下为您准备了快速运行的示例数据，若需要使用全量数据可以参考下方[效果复现](#效果复现)部分。
数据格式参见demo数据：data/train

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在esmm模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/multitask/esmm # 在任意目录均可运行
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 动态图预测
python -u ../../../tools/infer.py -m config.yaml 

# 静态图训练
python -u ../../../tools/static_trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 静态图预测
python -u ../../../tools/static_infer.py -m config.yaml 
```

## 模型组网
ESMM是发表在 SIGIR’2018 的论文[《Entire Space Multi-Task Model: An Eﬀective Approach for Estimating Post-Click Conversion Rate》](  https://arxiv.org/abs/1804.07931  )文章基于 Multi-Task Learning 的思路，提出一种新的CVR预估模型——ESMM，有效解决了真实场景中CVR预估面临的数据稀疏以及样本选择偏差这两个关键问题。模型的主要组网结构如下：
[ESMM](https://arxiv.org/abs/1804.07931):
<p align="center">
<img align="center" src="../../../doc/imgs/esmm.png">
<p>

### 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。 
在全量数据下模型的训练指标如下：
| 模型 | auc_ctr | batch_size | epoch_num | Time of each epoch |
| :------| :------ | :------ | :------| :------ | 
| ESMM | 0.82 | 1024 | 10 | 约3分钟 |

1. 确认您当前所在目录为PaddleRec/models/multitask/esmm  
2. 进入paddlerec/datasets/ali-ccp目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的ali-ccp全量数据集，并解压到指定文件夹。
``` bash
cd ../../../datasets/ali-ccp
sh run.sh
``` 
3. 切回模型目录,执行命令运行全量数据
```bash
cd - # 切回模型目录
# 动态图训练
python -u ../../../tools/trainer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml 
python -u ../../../tools/infer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml 
```

## 进阶使用
  
## FAQ
