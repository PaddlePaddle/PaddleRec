# wide&deep

以下是本例的简要目录结构及说明： 

```
├── data # 文档
	├── train #训练数据
		├── train_data.txt
	├── create_data.sh
	├── data_preparation.py
	├── get_slot_data.py
	├── run.sh
├── __init__.py 
├── config.yaml #配置文件
├── model.py #模型文件
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

## 内容

- [模型简介](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/wide_deep#模型简介)
- [数据准备](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/wide_deep#数据准备)
- [运行环境](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/wide_deep#运行环境)
- [快速开始](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/wide_deep#快速开始)
- [论文复现](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/wide_deep#论文复现)
- [进阶使用](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/wide_deep#进阶使用)
- [FAQ](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/wide_deep#FAQ)

## 模型简介

[《Wide & Deep Learning for Recommender Systems》]( https://arxiv.org/pdf/1606.07792.pdf)是Google 2016年发布的推荐框架，wide&deep设计了一种融合浅层（wide）模型和深层（deep）模型进行联合训练的框架，综合利用浅层模型的记忆能力和深层模型的泛化能力，实现单模型对推荐系统准确性和扩展性的兼顾。从推荐效果和服务性能两方面进行评价：

1. 效果上，在Google Play 进行线上A/B实验，wide&deep模型相比高度优化的Wide浅层模型，app下载率+3.9%。相比deep模型也有一定提升。
2. 性能上，通过切分一次请求需要处理的app 的Batch size为更小的size，并利用多线程并行请求达到提高处理效率的目的。单次响应耗时从31ms下降到14ms。

本例在paddlepaddle上实现wide&deep并在开源数据集Census-income Data上验证模型效果，在测试集上的平均acc和auc分别为：

> mean_acc: 0.76195
>
> mean_auc: 0.90577

若进行精度验证，请参考[论文复现](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/wide_deep#论文复现)部分。

本项目支持功能

训练：单机CPU、单机单卡GPU、单机多卡GPU、本地模拟参数服务器训练、增量训练，配置请参考 [启动训练](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/train.md)

预测：单机CPU、单机单卡GPU ；配置请参考[PaddleRec 离线预测](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/predict.md)

## 数据准备

数据地址： 

[adult.data](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)

[adult.test](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test)

## 运行环境

PaddlePaddle>=1.7.2

python 2.7/3.5/3.6/3.7

PaddleRec >=0.1

os : windows/linux/macos

## 快速开始

### 单机训练

CPU环境

在config.yaml文件中设置好设备，epochs等。

```sh
dataset:
  - name: sample_1
    type: QueueDataset
    batch_size: 5
    data_path: "{workspace}/data/sample_data/train"
    sparse_slots: "label"
    dense_slots: "wide_input:8 deep_input:58"
  - name: infer_sample
    type: QueueDataset
    batch_size: 5
    data_path: "{workspace}/data/sample_data/train"
    sparse_slots: "label"
    dense_slots: "wide_input:8 deep_input:58"              
```

### 单机预测

CPU环境

在config.yaml文件中设置好epochs、device等参数。

```
 - name: infer_runner
    class: infer
    device: cpu
    init_model_path: "increment/0"
```


## 论文复现


用原论文的完整数据复现论文效果需要在config.yaml中修改batch_size=40, thread_num=8, epoch_num=40

本例在paddlepaddle上实现wide&deep并在开源数据集Census-income Data上验证模型效果，在测试集上的平均acc和auc分别为：

mean_acc: 0.76195 , mean_auc: 0.90577


修改后运行方案：修改config.yaml中的'workspace'为config.yaml的目录位置，执行

```
python -m paddlerec.run -m /home/your/dir/config.yaml #调试模式 直接指定本地config的绝对路径
```

## 进阶使用

## FAQ
