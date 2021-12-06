# difm (A Dual Input-aware Factorization Machine for CTR Prediction)

代码请参考：[difm](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/difm)  
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
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。简单来说，CTR预估对每次广告的点击情况做出预测，预测用户是点击还是不点击。CTR预估模型综合考虑各种因素、特征，在大量历史数据上训练，最终对商业决策提供帮助。本模型实现了下述论文中的 DIFM 模型：

```text
@inproceedings{lu2020dual,
  title={A Dual Input-aware Factorization Machine for CTR Prediction.},
  author={Lu, Wantong and Yu, Yantao and Chang, Yongzhe and Wang, Zhen and Li, Chenhui and Yuan, Bo},
  booktitle={IJCAI},
  pages={3139--3145},
  year={2020},
  url={https://www.ijcai.org/Proceedings/2020/0434.pdf}
}
```

## 数据准备

训练及测试数据集选用 [Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/) 所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。
每一行数据格式如下所示：
```
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔，缺失特征用空格表示。测试集中```<label>```特征已被移除。  
在模型目录的data目录下为您准备了快速运行的示例数据，若需要使用全量数据可以参考下方[效果复现](#效果复现)部分。

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在 difm 模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/rank/difm # 在任意目录均可运行
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml # 全量数据运行 config_bigdata.yaml 
# 动态图预测
python -u ../../../tools/infer.py -m config.yaml 

# 静态图训练
python -u ../../../tools/static_trainer.py -m config.yaml # 全量数据运行 config_bigdata.yaml 
# 静态图预测
python -u ../../../tools/static_infer.py -m config.yaml 
``` 

## 模型组网

DIFM 模型的组网本质是一个二分类任务，代码参考 `net.py`。模型主要组成是 Embedding 层，Dual-FEN 层，Reweighting 层， FM 特征交叉层以及相应的分类任务的loss计算和auc计算。

![DIFM](https://tva1.sinaimg.cn/large/008i3skNly1gtffgzgk1bj30kq0e8wfz.jpg)

### 主要流程

上图为 DIFM 的网络结构图，paper 题目中所指的 Dual-FEN 为 `vector-wise` 和 `bit-wise`两个 Input-aware Factorization 模块, 一个是 bit-wise,
一个是 vector-wise。只是维度上不同，实现的直觉是一样的。bit-wise 维度会对某一个 sparse embedding 向量内部彼此进行交叉，而 vector-wise 仅仅处理
embedding 向量层次交叉。把 vector-wise FEN 模块去掉，DIFM 就退化为 IFM 模型了，该算法也是论文作者实验组的大作，其结构图如下：

![IFM](https://tva1.sinaimg.cn/large/008i3skNly1gtffi72287j60ez0cwq3p02.jpg)

两类不同维度的 FEN(Factor Estimating Net) 作用都是一致的，即输出 Embedding Layer 相应向量的权重。举个例子，假设上游有 n 个 sparse features， 
则 FEN 输出结果为 [a1, a2, ..., an]. 在 Reweighting Layer 中，对原始输入进行权重调整。最后输入到 FM 层进行特征交叉，输出预测结果。因此，总结两篇论文步骤如下：

- sparse features 经由 Embedding Layer 查表得到 embedding 向量，dense features 特征如何处理两篇论文都没提及；
- sparse features 对应的一阶权重也可以通过 1 维 Embedding Layer 查找；
- sparse embeddings 输入 FEN (bit-wise or vector-wise)，得到特征对应的权重 [a1, a2, ..., an]；
- Reweighting Layer 根据上一步骤中的特征权重，对 sparse embeddings 进一步调整；
- FM Layer 进行特征交叉，输出预测概率；


### Loss 及 Auc 计算
- 为了得到每条样本分属于正负样本的概率，我们将预测结果和 `1-predict` 合并起来得到 `predict_2d`，以便接下来计算 `auc`。  
- 每条样本的损失为负对数损失值，label的数据类型将转化为float输入。  
- 该batch的损失 `avg_cost` 是各条样本的损失之和
- 我们同时还会计算预测的auc指标。

## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现 README 中的效果,请按如下步骤依次操作即可。
在全量数据下模型的指标如下：  

| 模型 | auc | batch_size | epoch_num| Time of each epoch |
| :------| :------ | :------ | :------| :------ | 
| DIFM | 0.7999+ | 2000 | 2 | 约 7.5 小时 |

1. 确认您当前所在目录为 `PaddleRec/models/rank/difm`
2. 进入 `PaddleRec/datasets/criteo` 目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的 criteo 全量数据集，并解压到指定文件夹。
``` bash
cd ../../../datasets/criteo
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
