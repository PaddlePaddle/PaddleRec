# mmoe (Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts)

代码请参考：[MMOE](https://github.com/PaddlePaddle/PaddleRec/blob/master/models/multitask/mmoe)  
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
多任务模型通过学习不同任务的联系和差异，可提高每个任务的学习效率和质量。多任务学习的的框架广泛采用shared-bottom的结构，不同任务间共用底部的隐层。这种结构本质上可以减少过拟合的风险，但是效果上可能受到任务差异和数据分布带来的影响。  论文[《Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts》]( https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture- )中提出了一个Multi-gate Mixture-of-Experts(MMOE)的多任务学习结构。

## 数据准备
我们在开源数据集Census-income Data上验证模型效果,在模型目录的data目录下为您准备了快速运行的示例数据，若需要使用全量数据可以参考下方[效果复现](#效果复现)部分.
数据的格式如下：
生成的格式以逗号为分割点
```
0,0,73,0,0,0,0,1700.09,0,0
```

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在mmoe模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/multitask/mmoe # 在任意目录均可运行
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
MMOE模型刻画了任务相关性，基于共享表示来学习特定任务的函数，避免了明显增加参数的缺点。模型的主要组网结构如下：
[MMoE](https://dl.acm.org/doi/abs/10.1145/3219819.3220007):
<p align="center">
<img align="center" src="../../../doc/imgs/mmoe.png">
<p>

### 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。 
在全量数据下模型的指标如下：
| 模型 | auc_marital | batch_size | epoch_num | Time of each epoch |
| :------| :------ | :------ | :------| :------ | 
| MMOE | 0.99 | 32 | 100 | 约1分钟 |

1. 确认您当前所在目录为PaddleRec/models/multitask/mmoe  
2. 进入paddlerec/datasets/census目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的census全量数据集，并解压到指定文件夹。
``` bash
cd ../../../datasets/census
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
