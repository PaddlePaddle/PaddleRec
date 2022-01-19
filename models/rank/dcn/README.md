# 基于DeepAndCross模型的点击率预估模型

**[AI Studio在线运行环境](https://aistudio.baidu.com/aistudio/projectdetail/3240207)**

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── sample_data # 数据样例
        ├── sample_train.txt #训练数据样例
├── __init__.py
├── README.md #文档
├── net.py #组网文件
├── dygraph_model.py #动态图模型文件
├── static_model.py #静态图模型文件
├── config.yaml #样本数据训练推理配置文件
├── config_big.yaml #全量数据训练推理配置文件
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
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。简单来说，CTR预估对每次广告的点击情况做出预测，预测用户是点击还是不点击。CTR预估模型综合考虑各种因素、特征，在大量历史数据上训练，最终对商业决策提供帮助。本模型实现了下述论文中的DeepAndCross模型：

```text
@inproceedings{DeepAndCross,
  title={DeepAndCross: Deep & Cross Network for Ad Click Predictions},
  author={Ruoxi Wang, Bin Fu, Gang Fu, Mingliang Wang},
  year={2017}
}
```

## 数据准备
### 数据来源
训练及测试数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。
每一行数据格式如下所示：
```bash
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔，缺失特征用空格表示。测试集中```<label>```特征已被移除。  

### 一键下载训练及测试数据
全量数据集解析过程:
1. 确认您当前所在目录为PaddleRec/models/rank/dcn
2. 进入paddlerec/datasets/criteo目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的criteo全量数据集，并解压到指定文件夹。自动处理数据转化为可直接进行训练的格式。解压后全量训练数据放置于`./slot_train_data_full`，全量测试数据放置于`./slot_test_data_full`

``` bash
cd ../../../datasets/criteo
sh run.sh
``` 

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在paddlerec模型目录"PaddleRec/models/rank/dcn"目录下执行下面的命令即可快速启动训练： 

```bash
# 进入模型目录
# cd models/rank/dcn # 在任意目录均可运行
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 动态图预测
python -u ../../../tools/infer.py -m config.yaml 

# 静态图训练
python -u ../../../tools/static_trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 静态图预测
python -u ../../../tools/static_infer.py -m config.yaml 
```

注意训练-预测，模型存储文件位置的一致性；动态图模型与静态图模型存储位置要分开；

## 模型组网

deepAndCross模型的组网本质是一个二分类任务，模型代码参考“dygraph_model.py, static_model.py”，组网代码参考`net.py`。模型主要组成是交叉项Cross部分，DNN部分,以及相应的分类任务的loss、正则项loss计算，和auc计算。模型的组网可以看做Cross部分和dnn部分的结合，其中Cross部分主要的工作是通过特征间交叉得到交叉组合特征，可以实现任意特征间组合特征。dnn部分的主要组成为三个全连接层[512, 256, 128]，每层FC都后接一个relu激活函数，每层FC的初始化方式为符合正态分布的随机初始化.    
最后接了一层输出维度为128的fc层，其输出与Cross部分输出特征进行concat,经过一层fc,综合计算预测值。  

### Loss及Auc计算
- 预测的结果将的cross部分以及dnn部分输出的隐向量特征concat，再通过一层fc和激活函数sigmoid给出，为了得到每条样本分属于正负样本的概率，我们将预测结果和`1-predict`合并起来得到predict_2d，以便接下来计算auc。  
- 每条样本的损失为负对数损失值，label的数据类型将转化为float输入。  
- 该batch的损失`avg_cost`是各条样本的损失之和
- 我们同时还会计算预测的auc，auc的结果由`fluid.layers.auc()`给出，该层的返回值有三个，分别是全局auc: `auc_var`，当前batch的auc: `batch_auc_var`，以及auc_states: `_`，auc_states包含了`batch_stat_pos, batch_stat_neg, stat_pos, stat_neg`信息。
完成上述组网后，我们最终可以通过训练拿到`auc`指标。

## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。
在全量数据下模型的指标如下：  

| 模型 | auc | batch_size | epoch_num| Time of each epoch |
| :------| :------ | :------ | :------| :------ | 
| dcn |  0.777   | 32  |  10  | 约 3 小时 |

1. 确认您当前所在目录为PaddleRec/models/rank/dcn
2. 在"criteo data"全量数据目录下，运行数据一键处理脚本，命令如下：  
```bash
cd ../../../datasets/criteo
sh run.sh
```
3. 退回dcn目录中，配置改为使用config_bigdata.yaml中的参数  

4. 运行命令，模型会进行两个epoch的训练，然后预测第二个epoch，并获得相应auc指标  
```bash
cd - # 切回模型目录
# 动态图训练
python -u ../../../tools/trainer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml 
python -u ../../../tools/infer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml
```
注意训练-预测，模型存储文件位置的一致性；
## 进阶使用
  
## FAQ
