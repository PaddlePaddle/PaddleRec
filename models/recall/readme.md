# 召回模型库

## 简介
我们提供了常见的召回任务中使用的模型算法的PaddleRec实现, 单机训练&预测效果指标以及分布式训练&预测性能指标等。实现的召回模型包括 [SR-GNN](gnn)、[GRU4REC](gru4rec)、[Sequence Semantic Retrieval Model](ssr)、[Word2Vector](word2vec)、[Youtube_DNN](youtube_dnn)、[ncf](ncf)。

模型算法库在持续添加中，欢迎关注。

## 目录
* [整体介绍](#整体介绍)
    * [召回模型列表](#召回模型列表)
* [使用教程](#使用教程)
    * [训练 预测](#训练 预测)
* [效果对比](#效果对比)
    * [模型效果列表](#模型效果列表)

## 整体介绍
### 召回模型列表

|       模型        |       简介        |       论文        |
| :------------------: | :--------------------: | :---------: |
| Word2Vec | word2vector | [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)(2013) |
| GRU4REC | SR-GRU | [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)(2015) |
| Youtube_DNN | Youtube_DNN | [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)(2016) |
| SSR | Sequence Semantic Retrieval Model | [Multi-Rate Deep Learning for Temporal Recommendation](http://sonyis.me/paperpdf/spr209-song_sigir16.pdf)(2016) |
| NCF | Neural Collaborative Filtering | [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)(2017) |
| GNN | SR-GNN | [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855)(2018) |

下面是每个模型的简介（注：图片引用自链接中的论文）

[Word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf):
<p align="center">
<img align="center" src="../../doc/imgs/word2vec.png">
<p>

[GRU4REC](https://arxiv.org/abs/1511.06939):
<p align="center">
<img align="center" src="../../doc/imgs/gru4rec.png">
<p>

[Youtube_DNN](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf):
<p align="center">
<img align="center" src="../../doc/imgs/youtube_dnn.png">
<p>

[SSR](http://sonyis.me/paperpdf/spr209-song_sigir16.pdf):
<p align="center">
<img align="center" src="../../doc/imgs/ssr.png">
<p>

[NCF](https://arxiv.org/pdf/1708.05031.pdf):
<p align="center">
<img align="center" src="../../doc/imgs/ncf.png">
<p>

[GNN](https://arxiv.org/abs/1811.00855):
<p align="center">
<img align="center" src="../../doc/imgs/gnn.png">
<p>

## 使用教程
### 训练 预测
```shell
python -m paddlerec.run -m paddlerec.models.recall.word2vec # word2vec
python -m paddlerec.run -m paddlerec.models.recall.ssr # ssr
python -m paddlerec.run -m paddlerec.models.recall.gru4rec # gru4rec
python -m paddlerec.run -m paddlerec.models.recall.gnn # gnn
python -m paddlerec.run -m paddlerec.models.recall.ncf # ncf
python -m paddlerec.run -m paddlerec.models.recall.youtube_dnn # youtube_dnn
```
## 效果对比
### 模型效果列表

|       数据集        |       模型       |       HR@10        |       Recall@20       | 
| :------------------: | :--------------------: | :---------: |:---------: |
|       DIGINETICA     |       GNN       |       --        |       0.507       |
|       RSC15        |       GRU4REC       |       --        |       0.670          |
|       RSC15        |       SSR       |       --        |       0.590          |
|       MOVIELENS        |       NCF       |       0.688        |       --          |
|       --        |       Youtube       |       --        |       --          |
|       1 Billion Word Language Model Benchmark        |       Word2Vec       |       --         |       0.54          |

