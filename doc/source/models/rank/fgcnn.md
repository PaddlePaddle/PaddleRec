# fgcnn (Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction)

代码请参考：[fgcnn](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/fgcnn)  
如果我们的代码对您有用，还请点个star啊~  

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [效果复现](#效果复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)

## 模型简介
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。本模型实现了下述论文中提出的rank模型：

```text
@inproceedings{FGCNN,
  title={Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction},
  author={Bin Liu, Ruiming Tang, Yingzhi Chen, Jinkai Yu, Huifeng Guo, Yuzhou Zhang},
  year={2019}
}

Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, Xiuqiang He. [Open Benchmarking for Click-Through Rate Prediction](https://arxiv.org/abs/2009.05794). *The 30th ACM International Conference on Information and Knowledge Management (CIKM)*, 2021. [[Bibtex](https://dblp.org/rec/conf/cikm/ZhuLYZH21.html?view=bibtex)]

Jieming Zhu, Kelong Mao, Quanyu Dai, Liangcai Su, Rong Ma, Jinyang Liu, Guohao Cai, Zhicheng Dou, Xi Xiao, Rui Zhang. [BARS: Towards Open Benchmarking for Recommender Systems](https://arxiv.org/pdf/2205.09626.pdf). *The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)*, 2022. [Bibtex]
```

增加人工的特征通常会提升效果，但是人工设计特征代价很高。因此需要一种自动提取有效特征，丰富特征表示的方式。该工作提出了Feature Generation by Convolutional Neural Network (FGCNN)模型解决该问题。
FGCNN有两个模块： Feature Generation 和 Deep Classifier。
其中Feature Generation利用CNN去生成local patterns并且组合生成新的特征。
Deep Classifier则采用IPNN的结构去学习增强特征空间中的交互。
该工作表明CTR预测的一个新方向：通过外部的模型减少DNN部分学习高阶特征的难度，本文就是通过CNN+MLP学习的特征，添加到DNN部分。

## 数据准备
训练及测试数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。
每一行数据格式如下所示：
```
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔，缺失特征用空格表示。测试集中```<label>```特征已被移除。  
在模型目录的data目录下为您准备了快速运行的示例数据，若需要使用全量数据可以参考下方[效果复现](#效果复现)部分。

## 运行环境
PaddlePaddle>=2.1

python 3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在fgcnn模型目录的快速执行命令如下： 
```bash
# 进入模型目录
cd models/rank/fgcnn 
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml 
# 动态图预测
python -u ../../../tools/infer.py -m config.yaml 

```

## 效果复现
### 数据集获取及预处理
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。同时，我们提供了全量数据生成的脚本，将会自动下载转换好格式的criteo数据集。
在全量数据下模型的指标如下：  
| 模型 | auc | batch_size | epoch_num| Time of each epoch |
| :------| :------ | :------ | :------| :------ | 
| fgcnn |  0.8022   | 2000  |  2  | 约 2 小时 |

1. 确认您当前所在目录为PaddleRec/models/rank/fgcnn
2. 进入paddlerec/datasets/criteo_fgcnn目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的criteo全量数据集，并解压到指定文件夹。
``` bash
cd ../../../datasets/criteo_fgcnn
sh run.sh
``` 
3. 切回模型目录,执行命令运行全量数据
```bash
# 切回模型目录
cd -
# 动态图训练
python -u ../../../tools/trainer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml 
python -u ../../../tools/infer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml 
```

## 进阶使用
  
## FAQ
