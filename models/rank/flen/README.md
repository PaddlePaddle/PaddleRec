# 基于 FLEN 模型的点击率预估模型

以下是本例的简要目录结构及说明： 

```
├── data # 样例数据
    ├── sample_data # 样例数据
        ├── train
            ├── sample_train.txt # 训练数据样例
├── __init__.py
├── README.md # 文档
├── config.yaml # sample数据配置
├── config_bigdata.yaml # 全量数据配置
├── net.py # 模型核心组网（动静统一）
├── avazu_reader.py # 数据读取程序
├── dygraph_model.py # 构建动态图
```

注：在阅读该示例前，建议您先了解以下内容：

[PaddleRec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

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
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。简单来说，CTR预估对每次广告的点击情况做出预测，预测用户是点击还是不点击。CTR预估模型综合考虑各种因素、特征，在大量历史数据上训练，最终对商业决策提供帮助。本模型实现了下述论文中的 FLEN 模型：

```text
@article{FLEN2020,
  title={FLEN: Leveraging Field for Scalable CTR Prediction},
  author={Wenqiang Chen, Lizhang Zhan, Yuanlong Ci, Minghua Yang, Chen Lin, Dugang Liu},
  journal={arXiv preprint arXiv:1911.04690},
  year={2020}，
  url={https://arxiv.org/abs/1911.04690},
}
```

## 数据准备

训练及测试数据集选用[Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction)所用的Avazu数据集。该数据集包括10天时间段的点击数据，包含两部分：训练集和测试集。训练集包含前9天内Avazu的点击流量，测试集则对应训练数据后一天的点击流量。
每一行数据格式如下所示：
```
<label> <categorical feature 1> ... <categorical feature 22>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<categorical feature>```代表分类特征（离散特征），共有22个离散特征。相邻两个特征用```，```分隔，缺失特征用'-1'表示。测试集中```<label>```特征已被移除。  
在模型目录的data目录下为您准备了快速运行的示例数据，若需要使用全量数据可以参考下方[效果复现](#效果复现)部分。

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始


本文提供了[FLEN-Paddle AiStudio项目](https://aistudio.baidu.com/aistudio/projectdetail/3247609)可以供您快速体验，进入项目快速开始。


## 模型组网

FLEN 模型的组网，代码参考 `net.py`。模型主要组成是 Embedding层，FieldWiseBiInteraction 层，MLP 层，以及相应的分类任务的loss计算和auc计算。模型架构如下：

<img align="center" src="https://wx2.sinaimg.cn/mw2000/0073e4AWly1gxnk6tahgpj30dl0f2ae8.jpg">

### **Embedding 层**

FLEN 模型的特征输入，主要包括 sparse 类别特征。sparse features 经由 Embedding 层查找得到相应的 embedding 向量。同时，将Sparse特征根据语义划分为三个不
同Fields，最终得到三个Fields的特征嵌入表示。Embedding层的网络结构如下图所示：

<img align="center" src="https://wx1.sinaimg.cn/mw2000/0073e4AWly1gxnk9e1pgoj30dv05d762.jpg">


### **FieldWiseBiInteraction层**
FieldWiseBiInteraction层主要包含两个部分，FM模块和MF模块。特征经过Embedding层后分别输入到FM和MF部分，再将两部分的输出进行SumPooling输出，计算公式如下所示：

<img align="center" src="https://wx4.sinaimg.cn/mw2000/0073e4AWly1gxnke0ksdhj30dr08oq56.jpg">



### **Loss 及 Auc 计算**
- 为了得到每条样本分属于正负样本的概率，我们将预测结果和 `1-predict` 合并起来得到 `predict_2d`，以便接下来计算 `auc`。  
- 每条样本的损失为负对数损失值，label的数据类型将转化为float输入。  
- 该batch的损失 `avg_cost` 是各条样本的损失之和
- 我们同时还会计算预测的auc指标。

## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现 README 中的效果,请按如下步骤依次操作即可。
在全量数据下模型的指标如下：  

| 模型 | auc | logloss | batch_size | epoch_num| Time of each epoch |
| :------| :------ | :------ | :------| :------ | :------ | 
| FLEN | 0.7516 | 0.3963 |512 | 1 | 约 1 小时 |

1. 确认您当前所在目录为PaddleRec/models/rank/flen
2. 进入paddlerec/datasets/Avazu_flen目录下，根据readme.md获取数据，您可以下载原始数据放到指定文件夹，执行命令得到训练集
和测试集，或者在[AiStudio数据集](https://aistudio.baidu.com/aistudio/datasetdetail/125200)获取已经预处理好的训练集和测试集。
``` bash
cd ../../../datasets/Avazu_flen
# 处理数据、划分数据集大约耗时35分钟
sh data_process.sh
``` 
3. 确认您当前所在目录为 `PaddleRec/models/rank/flen`

```
# 动态图训练
python -u ../../../tools/trainer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml 
python -u ../../../tools/infer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml 
```

## 进阶使用
  
## FAQ
