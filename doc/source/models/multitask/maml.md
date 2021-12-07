# maml (Model-agnostic meta-learning for fast adaptation of deep networks)

代码请参考：[MAML](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/maml)  
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
[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400.pdf)算法是一种模型无关的元学习算法，其模型无关体现在，能够与任何使用了梯度下降法的模型相兼容，广泛应用于各种不同的机器学习任务，包括分类、识别、强化学习等领域。  
元学习的目标，是在大量不同的任务上训练一个模型，使其能够使用极少量的训练数据（即小样本），进行极少量的梯度下降步数，就能够迅速适应新任务，解决新问题。  
本模型来自飞桨论文复现挑战赛 hrdws 大神贡献的[MAML元学习算法，小样本学习，多任务学习](https://aistudio.baidu.com/aistudio/projectdetail/1869590?channelType=0&channel=0)。

## 数据准备
训练及测试数据集选用omniglot数据集。Omniglot 数据集包含50个不同的字母表，每个字母表中的字母各包含20个手写字符样本，每一个手写样本都是不同的人通过亚马逊的 Mechanical Turk 在线绘制的。Omniglot数据集的多样性强于MNIST数据集，是增强版的MNIST，常用与小样本识别任务。  
在模型目录的data目录下为您准备了快速运行的示例数据，若需要使用全量数据可以参考下方[效果复现](#效果复现)部分。

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 


## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在maml模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/multitask/maml # 在任意目录均可运行
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 动态图预测
python -u ../../../tools/infer.py -m config.yaml 
```

## 模型组网
考虑一个关于任务T的分布p(T)，我们希望模型能够对该任务分布很好的适配。在K-shot（即K个学习样本）的学习任务下，从p(T)分布中随机采样一个新任务Ti，在任务Ti的样本分布qi中随机采样K个样本，用这K个样本训练模型，获得LOSS，实现对模型f的内循环更新。然后再采样query个样本，评估新模型的LOSS，然后对模型f进行外循环更新。反复上述过程，从而使最终模型能够对任务分布p(T)上的所有情况，能够良好地泛化。MAML算法针对小样本图像分类任务的计算流程可用下图进行示意:  
<p align="center">
<img align="center" src="../../../doc/imgs/maml.png">
<p>

## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。
在全量数据下模型的指标如下：  

| 模型 | acc | batch_size | thread_num| epoch_num| Time of each epoch |
| :------| :------ | :------| :------ | :------| :------ | 
| maml | 0.98 | 32 | 1 | 100 | 约4分钟 |

1. 确认您当前所在目录为PaddleRec/models/multitask/maml
2. 进入paddlerec/datasets/omniglot目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的Omniglot全量数据集，并解压到指定文件夹。
``` bash
cd ../../../datasets/omniglot
sh run.sh
``` 
3. 切回模型目录,执行命令运行全量数据
```bash
cd - # 切回模型目录
export FLAGS_cudnn_deterministic=True # 固定确定性算法
# 动态图训练
python -u ../../../tools/trainer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml 
python -u ../../../tools/infer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml 
```

## 进阶使用
  
## FAQ
