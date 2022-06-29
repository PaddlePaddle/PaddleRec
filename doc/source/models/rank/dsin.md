# dsin (Deep Session Interest Network for Click-Through Rate Prediction)

代码请参考：[DSIN](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/dsin)  
如果我们的代码对您有用，还请点个star啊~  


## 内容

- [DSIN模型](#dsin模型)
- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [模型组网](#模型组网)
- [效果复现](#效果复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)

## 模型简介
模型主要聚焦于用户的历史会话行为，通过Self-Attention和BiLSTM对历史会话行为进行学习，最后通过Activation Unit得到最终的session表征向量，再结合其他特征送入MLP计算最后的ctr score。[Deep Session Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.06482v1.pdf)文章通过 Transformer 和 BiLSTM 来学习用户的 Session Interest Interacting，提升模型的表达能力。[知乎解析看这里](https://zhuanlan.zhihu.com/p/514780690)

## 数据准备
本模型使用论文中的数据集Alimama Dataset，参考[原文作者的数据预处理过程](https://github.com/shenweichen/DSIN/tree/master/code)对数据进行处理。

## 运行环境
PaddlePaddle == 2.2.2

python 3.7.4

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在DMR模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/rank/dsin # 在任意目录均可运行
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
论文[Deep Session Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.06482v1.pdf)中的网络结构如图所示:  
<p align="center">
<img align="center" src="../../../doc/imgs/dsin.png">
<p>

## 效果复现 
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。  
在全量数据下模型的指标如下：

| 模型 | auc | batch_size | epoch_num | Time of each epoch |
| :------| :------ | :------ | :------| :------ | 
| DSIN | 0.6356 | 4096 | 1 | 约10分钟 |

1. 确认您当前所在目录为PaddleRec/models/rank/dsin  
2. 进入paddlerec/datasets/Ali_Display_Ad_Click_DSIN目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的Alimama全量数据集，并解压到指定文件夹。若您希望从原始数据集自行处理，请详见该目录下的readme。

``` bash
cd ../../../datasets/Ali_Display_Ad_Click_DSIN
sh run.sh
```
3. 切回模型目录,执行命令运行全量数据

```bash
cd - # 切回模型目录
# 动态图训练
python -u ../../../tools/trainer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml 
python -u ../../../tools/infer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml
```

效果复现过程可参考[AI Studio项目](https://aistudio.baidu.com/aistudio/projectdetail/3850087)。

Note:运行环境为至尊GPU。

## 进阶使用
  
## FAQ
