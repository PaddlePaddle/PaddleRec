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

## 效果复现
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

