# 基于 FAT_DeepFFM 模型的点击率预估模型

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
├── criteo_reader.py # 数据读取程序
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
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。简单来说，CTR预估对每次广告的点击情况做出预测，预测用户是点击还是不点击。CTR预估模型综合考虑各种因素、特征，在大量历史数据上训练，最终对商业决策提供帮助。本模型实现了下述论文中的 FAT_DeepFFM 模型：

```text
@article{FAT-DeepFFM2019,
  title={FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine},
  author={Junlin Zhang, Tongwen Huang, Zhiqi Zhang},
  journal={arXiv preprint arXiv:1905.06336},
  year={2019}，
  url={https://arxiv.org/pdf/1905.06336},
}
```

## 数据准备

训练及测试数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。
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
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在fat_deepffm模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/rank/fat_deepffm # 在任意目录均可运行
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

FAT_DeepFFM 模型的组网，代码参考 `net.py`。模型主要组成是 Embedding 层，CENet 层，DeepFFM特征交叉层，DNN层以及相应的分类任务的loss计算和auc计算。模型架构如下：

<img align="center" src="https://wx4.sinaimg.cn/orj360/0073e4AWly1gx7mmb6m3oj30rq0g1tf2.jpg">

### **CENet 层**

FAT_DeepFFM 模型的特征输入，主要包括 sparse 类别特征。（在处理 dense 数值型特征时，进行升维与sparse 类别特征拼接）
sparse features 经由 embedding 层查找得到相应的 embedding 向量。使用CENet显示地建模特征之间的依赖关系。CENet网络结构如下图所示：

<img align="center" src="https://wx1.sinaimg.cn/bmiddle/0073e4AWly1gx7okwknn4j30so0fcq6u.jpg">

根据网络结构图，通过CENet的注意力机制有选择性地突出信息特征并抑制不太有用的特征，公式如下所示：

<img align="center" src="https://wx4.sinaimg.cn/mw2000/0073e4AWly1gx7oscgpc9j30ev01s3z2.jpg">


### **DeepFFM层**
DeepFFM网络结构如下图所示：

<img align="center" src="https://wx1.sinaimg.cn/orj360/0073e4AWly1gx7p0casynj30qk0fq461.jpg">

使用FFM对特征的不同field的关系进行建模，计算如下公式所示：

<img align="center" src="https://wx1.sinaimg.cn/mw2000/0073e4AWly1gx7p2f1t2cj30e1024t99.jpg">



### **Loss 及 Auc 计算**
- 为了得到每条样本分属于正负样本的概率，我们将预测结果和 `1-predict` 合并起来得到 `predict_2d`，以便接下来计算 `auc`。  
- 每条样本的损失为负对数损失值，label的数据类型将转化为float输入。  
- 该batch的损失 `avg_cost` 是各条样本的损失之和
- 我们同时还会计算预测的auc指标。

## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现 README 中的效果,请按如下步骤依次操作即可。
在全量数据下模型的指标如下：  

| 模型 | auc | batch_size | epoch_num| Time of each epoch |
| :------| :------ | :------ | :------| :------ | 
| FAT_DeepFFM | 0.8037 | 1000 | 1 | 约 3.5 小时 |

1. 确认您当前所在目录为 `PaddleRec/models/rank/fat_deepffm`
2. 进入 `PaddleRec/datasets/criteo` 目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的criteo全量数据集，并解压到指定文件夹。
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
