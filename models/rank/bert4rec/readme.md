# BERT4Rec模型

以下是本例的简要目录结构及说明： 

```
├── data #数据
    ├── beauty.txt #beauty数据集
├── readme.md #文档
├── config_bigdata.yaml # 全量数据配置
├── net.py # 模型核心组网
├── data_reader.py #数据读取程序
├── dygraph_model.py # 构建动态图
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [模型组网](#模型组网)
- [效果复现](#效果复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)


## 模型简介
BERT4Rec将NLP中的mask language任务迁移到序列推荐问题来，**给予了序列推荐一种不同于item2item，left2right的训练范式**。
具体来说，对于一条物品序列，**以一定的概率p随机mask掉序列中的物品**，**使用transformer的encoder结构**对mask item进行预测。
通过数据增强，完形填空任务的方式使得训练更加充分。

[BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://dl.acm.org/doi/abs/10.1145/3357384.3357895)
本模型来自飞桨论文复现挑战赛（第四期）的[BERT4Rec的第一名方案](https://aistudio.baidu.com/aistudio/projectdetail/2558070)达到了原作者github[BERT4Rec](https://github.com/FeiSun/BERT4Rec)的精度。

## 数据准备
本模型使用论文中的数据集Beauty Dataset，在模型目录的data目录下。

## 运行环境
PaddlePaddle>=2.0

python 3.7

## 模型组网
在BERT4Rec之前，SASRec已经将self-attention应用在了序列推荐任务中。

而与之对应的，BERT4Rec的作者认为像SASRec那种left-to-right的建模方式限制了模型的表达能力。 虽然用户的行为序列长期来看存在顺序的依赖关系，但在短期的一段时间内，用户的行为顺序不应该是严格顺序依赖的。

为了解决上述问题，BERT4Rec将NLP中的Mask Language任务迁移到序列推荐问题来，给予了序列推荐一种不同于item2item，left2right的训练范式。 
具体来说，对于一条物品序列，以一定的概率p随机mask掉序列中的物品（体现在data_augment_candi_gen.py)，使用Transformer的Encoder结构对masked item进行预测(net.py)。
训练的过程中，则是取出Encoder对应mask位置的representation来预测mask的label物品。即完形填空。

可以注意到，通过随机mask，我们可以成倍的生成新样本。（BERT4Rec训练样本是原有的十一倍(10:随机mask 1：原本样本)） 因此BERT4Rec效果的提升，也从数据增强的角度来解释。即通过数据增强，和完形填空式的前置任务的方式使得模型训练得更加充分。

在模型上面，BERT4Rec正如其名，就是跟BERT一样，使用Transformer的Encoder部分来做序列特征提取。

在测试的时候，我们只需要mask掉序列最后的物品，并取出模型最后一步的representation出来，那么就将训练好的模型成功应用在Next-item Prediction任务上。

## 效果复现
| 模型 | HR@10 | NDCG@10 | MRR | epoch_num| Time of each epoch |
| :------| :------ |:------ | :------ | :------| :------ | 
| BERT4Rec | 0.306 | 0.187 | 0.170 | 50 | 约2小时 |

本文提供了beauty数据集可以供您快速体验及其复现。在BERT4Rec模型目录的快速执行命令如下： 

```bash
# 进入模型目录
# cd models/rank/bert4rec # 在任意目录均可运行
# 数据增强与候选集生成
python -u data_augment_candi_gen.py 
# 动态图训练
python -u ../../../tools/trainer.py -m config_bigdata.yaml 
# 动态图预测
python -u ../../../tools/infer.py -m config_bigdata.yaml 

# 静态图训练

# 静态图预测

``` 

## 进阶使用
  
## FAQ

