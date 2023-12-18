# 基于Fastformer模型的点击率预估模型

**[AI Studio在线运行环境](https://aistudio.baidu.com/aistudio/projectdetail/3240529)**

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── sample_data #样例数据
        ├── train
            ├── sample_train.txt #训练数据样例
├── __init__.py
├── README.md #文档
├── config.yaml # sample数据配置
├── config_bigdata.yaml # 全量数据配置
├── fastformer.py # Fastformer attention 核心
├── net.py # 模型核心组网（动静统一）
├── FastformerDataReader.py #数据读取程序
├── dygraph_model.py # 构建动态图
├── static_model.py #构建静态图
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

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
@inproceedings{
  title={Fastformer: Additive Attention Can Be All You Need},
  author={Chuhan Wu, Fangzhao Wu, Tao Qi, Yongfeng Huang, Xing Xie},
  year={2021}
}
```

Fastformer是一种基于加性注意力的高效Transformer模型。Fastformer不是建模标记之间的成对交互作用，而是首先利用加性注意力机制来捕捉全局上下文，然后根据标记与全局上下文表示之间的交互作用，进一步转换每个标记的表示。通过这种方式，Fastformer能够实现具有线性复杂性的有效上下文建模。在过去，NAML实现了一个news-encoder，通过文本卷积提取文章特征，并采用注意力机制将特征压缩为一个n维向量（文章嵌入）。用户浏览过的n篇文章的文章嵌入向量组将再次通过注意力机制被进一步压缩成最终的用户行为嵌入（包含了用户行为特征）。此user-behavior-embedding和一篇新文章的文章嵌入的向量内积则表示用户对此文章的喜好程度。

在这里，我们继承了NAML的架构，并将Fastformer attention引入其中，取得了在测试数据上令人满意的结果。


## 数据准备
此模型训练和预测涉及用户浏览文章历史，以及文章的具体信息，需要先收集所有训练和预测数据里出现过的文章，
每篇文章用一行表示，存放在一个或多个以article{number}.txt为后缀的文件里，如article.txt, article3.txt
每行的格式为：
文章id 主类id 子类id 分词后的文章标题id 分词后的文章单词id
以上5项用tab符号分割，id均为自然数，分词后的文章标题id 和 分词后的文章单词id 都用空格做分隔符
另外还需要收集用户的浏览记录，存放在一个或多个以browse{number}.txt为后缀的文件里，如browse.txt, browse3.txt
每个用户的单次浏览序列用一行表示，格式为：
浏览过的文章id序列 接下来浏览过的文章id 接下来没浏览的文章id序列
以上3项用tab符号分割，id序列之间用空格分割，接下来没浏览的文章id序列如果没有实际数据，可以采用负采样生成,
但是没浏览的序列id个数建议大于等于yaml配置文件中的neg_candidate_size

在模型目录的data/sample_data目录下为您准备了快速运行的示例数据

## 运行环境
PaddlePaddle>=2.0

python 3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在naml模型目录的快速执行命令如下： 
```bash
# 进入模型目录
cd models/rank/naml 
# 动态图训练
python -u trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 动态图预测
python -u ../../../tools/infer.py -m config.yaml 
```
其中yaml文件的超参数解释如下：
  article_content_size: 每篇文章包含的最大单词（超过则截断）
  article_title_size:  每篇文章标题包含的最大单词（超过则截断）
  browse_size: 用户的最大浏览序列长度（超过则截断）
  neg_condidate_sample_size: 负采样个数
  word_dimension: 每个单词的embedding维度
  category_size: 主类个数（若这个数字为n，则数据中的category范围为【0，n-1】）
  sub_category_size: 副类个数（若这个数字为n，则数据中的category范围为【0，n-1】）
  category_dimension: 每个类别的embedding维度
  word_dict_size: 单词字典的大小（单词词典中建议留一个unk，表示统计过程中未出现过的单词应该映射到哪个id）

#### Loss及Acc计算
- 预测的结果为一个softmax向量，表示实际浏览文章和负采样文章同时出现的情况下被用户浏览的概率
- 样本的损失函数值由交叉熵给出
- 我们同时还会计算预测的auc

## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。
同时，我们提供了全量数据生成的脚本，将会自动下载microsoft news dataset全量数据集并转换为模型能接受的
输入格式，执行方法如下：

1.进入路径PaddleRec/datasets/MIND/data

2.执行 sh run.sh

3.脚本运行完成后，打开dict/yaml_info.txt，将其中的词向量大小，类目大小，子类目大小信息copy到config_bigdata.yaml
里，替换最后3行的超参数
  category_size
  sub_category_size  
  word_dict_size

4.运行：
```bash
python -u trainer.py -m config_bigdata.yaml
```
以下为训练2个epoch的结果
| 模型 | auc | batch_size | epoch_num| Time of each epoch| 
| :------| :------ | :------ | :------| :------ | 
| naml | 0.66 | 50 | 3 | 约4小时 | 
| fastformer | 0.72 | 32 | 10 | 约120小时 | 

预测
```bash
python -u ../../../tools/infer.py -m config_bigdata.yaml
```

期待预测auc为0.729


单机多卡执行方式(以训练为例)
python -m paddle.distributed.launch trainer.py -m config_bigdata.yaml
在此情况下将使用单机上所有gpu卡，若需要指定部分gpu卡执行，可以通过设置环境变量CUDA_VISIBLE_DEVICES
来实现。例如单机上有8张卡，只打算用前4卡张训练，可以设置export CUDA_VISIBLE_DEVICES=0,1,2,3
再执行训练脚本即可。
