# PaddleRec 基于 Movielens 数据集的全流程示例

## 模型的详细教程可以查阅： [十分钟！全流程！从零搭建推荐系统](https://aistudio.baidu.com/aistudio/projectdetail/559336)

## 本地运行流程

在本地需要安装`PaddleRec`及`PaddlePaddle`，推荐在`Linux` + `python2.7` 环境下执行此demo

本地运行流程与AiStudio流程基本一致，细节略有区别

### 离线训练
```shell
sh train.sh
```

### 离线测试
```shell
sh offline_test.sh
```

### 模拟在线召回
```shell
sh online_recall.sh
```

### 模拟在线排序
```shell
sh online_rank.sh
```
