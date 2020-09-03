# Youtebe-DNN

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
	├── train
		├── data.txt
    ├── test
		├── data.txt
├── generate_ramdom_data # 随机训练数据生成文件
├── __init__.py
├── README.md # 文档
├── model.py #模型文件
├── config.yaml #配置文件
├── data_prepare.sh #一键数据处理脚本
├── reader.py #reader
├── infer.py # 预测程序
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)


---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [论文复现](#论文复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)

## 模型简介
[《Deep Neural Networks for YouTube Recommendations》](https://link.zhihu.com/?target=https%3A//static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf) 这篇论文是google的YouTube团队在推荐系统上DNN方面的尝试，是经典的向量化召回模型，主要通过模型来学习用户和物品的兴趣向量，并通过内积来计算用户和物品之间的相似性，从而得到最终的候选集。YouTube采取了两层深度网络完成整个推荐过程：

1.第一层是**Candidate Generation Model**完成候选视频的快速筛选，这一步候选视频集合由百万降低到了百的量级。

2.第二层是用**Ranking Model**完成几百个候选视频的精排。

本项目在paddlepaddle上完成YouTube dnn的召回部分Candidate Generation Model，分别获得用户和物品的向量表示，从而后续可以通过其他方法（如用户和物品的余弦相似度）给用户推荐物品。

由于原论文没有开源数据集，本项目随机构造数据验证网络的正确性。

本项目支持功能

训练：单机CPU、单机单卡GPU、本地模拟参数服务器训练、增量训练，配置请参考 [启动训练](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/train.md)   

预测：单机CPU、单机单卡GPU；配置请参考[PaddleRec 离线预测](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/predict.md) 

## 数据处理
调用python generate_ramdom_data.py生成随机训练数据，每行数据格式如下：
```
#watch_vec;search_vec;other_feat;label
0.01,0.02,...,0.09;0.01,0.02,...,0.09;0.01,0.02,...,0.09;20
```
方便起见，我们提供了一键式数据生成脚本：
```
sh data_prepare.sh
```

## 运行环境

PaddlePaddle>=1.7.2 

python 2.7/3.5/3.6/3.7

PaddleRec >=0.1

os : windows/linux/macos

## 快速开始

### 单机训练

```
mode: [cpu_single_train]

runner:
- name: cpu_single_train
  class: train
  device: cpu   # if use_gpu, set it to gpu
  epochs: 20
  save_checkpoint_interval: 1
  save_inference_interval: 1
  save_checkpoint_path: "increment_youtubednn"
  save_inference_path: "inference_youtubednn"
  save_inference_feed_varnames: ["watch_vec", "search_vec", "other_feat"] # feed vars of save inference
  save_inference_fetch_varnames: ["l3.tmp_2"]
  print_interval: 1
```

### 单机预测
通过计算每个用户和每个物品的余弦相似度，给每个用户推荐topk视频：

cpu infer:
```
python infer.py --test_epoch 19 --inference_model_dir ./inference_youtubednn --increment_model_dir ./increment_youtubednn --watch_vec_size 64 --search_vec_size 64 --other_feat_size 64 --topk 5
```

gpu infer:
```
python infer.py --use_gpu 1 --test_epoch 19 --inference_model_dir ./inference_youtubednn --increment_model_dir ./increment_youtubednn --watch_vec_size 64 --search_vec_size 64 --other_feat_size 64 --topk 5
```
### 运行
```
python -m paddlerec.run -m paddlerec.models.recall.w2v
```

### 结果展示

样例数据训练结果展示：

```
Running SingleStartup.
Running SingleRunner.
batch: 1, acc: [0.03125]
batch: 2, acc: [0.0625]
batch: 3, acc: [0.]
...
epoch 0 done, use time: 0.0605320930481, global metrics: acc=[0.]
...
epoch 19 done, use time: 0.33447098732, global metrics: acc=[0.]
```

样例数据预测结果展示:
```
user:0, top K videos:[40, 31, 4, 33, 93]
user:1, top K videos:[35, 57, 58, 40, 17]
user:2, top K videos:[35, 17, 88, 40, 9]
user:3, top K videos:[73, 35, 39, 58, 38]
user:4, top K videos:[40, 31, 57, 4, 73]
user:5, top K videos:[38, 9, 7, 88, 22]
user:6, top K videos:[35, 73, 14, 58, 28]
user:7, top K videos:[35, 73, 58, 38, 56]
user:8, top K videos:[38, 40, 9, 35, 99]
user:9, top K videos:[88, 73, 9, 35, 28]
user:10, top K videos:[35, 52, 28, 54, 73]
```

## 进阶使用

## FAQ
